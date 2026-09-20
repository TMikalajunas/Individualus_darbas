"""Atkuriamas kraujo donorystės eksperimentas ir top-k kvietimų atranka."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import platform
import time
import urllib.request
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import t, ttest_rel, wilcoxon
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
    StratifiedKFold, StratifiedGroupKFold, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 2026
URL = 'https://archive.ics.uci.edu/static/public/176/blood+transfusion+service+center.zip'
DATA_SHA256 = '96c8e1091b9c037bcaf25a19b24b49d07771cd88689ffd273e056e9e8845ffe7'
NAMES = {'frequency': 'Bendras dažnis', 'logistic': 'Logistinė regresija',
         'svm': 'SVM RBF Platt', 'mlp': 'MLP'}


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')


def load_data():
    path = ROOT / 'data' / 'transfusion.data'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            name = next(n for n in z.namelist() if n.endswith('transfusion.data'))
            path.write_bytes(z.read(name))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != DATA_SHA256:
        raise ValueError('Duomenų kontrolinė suma pasikeitė; patikrinkite šaltinį')
    df = pd.read_csv(path)
    df.columns = ['R', 'F', 'M', 'T', 'y']
    assert df.shape == (748, 5) and set(df.y) == {0, 1}
    assert (df.M == 250 * df.F).all()
    return df, checksum


class Features(BaseEstimator, TransformerMixin):
    """M pašalinamas; F/T pasirenkamas tik jautrumo bandymui."""
    def __init__(self, keep_m=False, ratio=False):
        self.keep_m = keep_m
        self.ratio = ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = ['R', 'F', 'M', 'T'] if self.keep_m else ['R', 'F', 'T']
        out = X[cols].to_numpy(dtype=float)
        if self.ratio:
            out = np.column_stack([out, X['F'].to_numpy() / np.maximum(X['T'].to_numpy(), 1)])
        return out


def pipeline(model):
    return Pipeline([('features', Features()), ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()), ('model', model)])


def specification(name):
    if name == 'frequency':
        return pipeline(DummyClassifier(strategy='prior')), {}
    if name == 'logistic':
        return pipeline(LogisticRegression(max_iter=2000, random_state=SEED)), {
            'model__C': [0.01, 0.1, 1, 10], 'model__class_weight': [None, 'balanced']}
    if name == 'mlp':
        return pipeline(MLPClassifier(random_state=SEED, max_iter=600,
            early_stopping=False, n_iter_no_change=20)), {
            'model__hidden_layer_sizes': [(8,), (16,)], 'model__alpha': [0.01, 0.1],
            'model__learning_rate_init': [0.001, 0.01]}
    # Pipeline viduje kalibratorius: kiekviena imputacija ir z-norma išmokstama
    # tik kalibravimo CV mokymo dalyje. ensemble=False palieka vieną galutinį SVC.
    model = CalibratedClassifierCV(pipeline(SVC(kernel='rbf', probability=False,
        random_state=SEED)), method='sigmoid',
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED+2), ensemble=False)
    return model, {'estimator__model__C': [0.1, 1, 10],
        'estimator__model__gamma': ['scale', 0.1],
        'estimator__model__class_weight': [None, 'balanced']}


def rank_indices(p, ids):
    # Lygių tikimybių atveju naudoti ID, o ne žymes ar originalų failo eiliškumą.
    tie = np.random.default_rng(SEED + 7).permutation(748)[np.asarray(ids, dtype=int)]
    return np.lexsort((tie, -np.asarray(p)))


def precision_at_k(y, p, ids, k):
    if not 1 <= k <= len(y):
        raise ValueError('k turi būti nuo 1 iki donorų skaičiaus')
    return float(np.asarray(y)[rank_indices(p, ids)[:k]].mean())


def metrics(y, p, ids):
    return {'ap': average_precision_score(y, p), 'brier': brier_score_loss(y, p),
        **{f'p{q}': precision_at_k(y, p, ids, math.ceil(len(y)*q/100)) for q in [10,20,30]}}


def predict_decision(fitted_pipeline, X):
    """Plano (1),(3): g(x)=sum alpha_i*y_i*exp(-gamma||s_i-z||²)+b.

    Ženklas duoda klasę; nepasirašyta g(x) reikalinga Platt kalibravimui.
    dual_coef_ jau apima alpha_i*y_i. Taikoma standartizuotiems požymiams.
    """
    z = fitted_pipeline[:-1].transform(X)
    svc = fitted_pipeline.named_steps['model']
    distances = ((z[:, None, :] - svc.support_vectors_[None, :, :])**2).sum(axis=2)
    kernel = np.exp(-svc._gamma * distances)
    return kernel @ svc.dual_coef_[0] + svc.intercept_[0]


def fit_search(name, X, y, cv):
    estimator, grid = specification(name)
    search = GridSearchCV(estimator, grid, cv=cv, scoring='average_precision',
        refit=True, n_jobs=1, error_score='raise', return_train_score=False)
    search.fit(X, y)
    return search


def bootstrap(y, predictions, ids, count=2000):
    rng = np.random.default_rng(SEED+9)
    rows = {name: [] for name in predictions}
    delta = []
    y = np.asarray(y)
    for _ in range(count):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        for name, p in predictions.items():
            rows[name].append(metrics(y[ix], p[ix], np.asarray(ids)[ix]))
        delta.append(rows['svm'][-1]['ap']-rows['logistic'][-1]['ap'])
    result = {name: {m: np.quantile([r[m] for r in values], [0.025,0.975]).tolist()
        for m in values[0]} for name, values in rows.items()}
    result['svm_minus_logistic_ap'] = np.quantile(delta, [0.025,0.975]).tolist()
    return result


def run():
    started = time.perf_counter()
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (ROOT / 'models').mkdir(exist_ok=True)
    df, checksum = load_data()
    ids = np.arange(len(df))
    dev, test = train_test_split(ids, test_size=0.2, stratify=df.y, random_state=SEED)
    X, y = df.drop(columns='y'), df.y
    # Galutinis testas čia tik užfiksuojamas; perskaitomas po visų dev eksperimentų.
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
        random_state=SEED+1).split(X.iloc[dev], y.iloc[dev]))
    save_json(out/'splits.json', {'seed': SEED, 'dev': dev.tolist(), 'test':test.tolist(),
        'outer': [{'train': dev[a].tolist(), 'validation': dev[b].tolist()} for a,b in folds]})
    quality = {'shape':list(df.shape), 'positives':int(y.sum()), 'prevalence':y.mean(),
        'missing':df.isna().sum().to_dict(), 'duplicate_rows':int(df.duplicated().sum()),
        'duplicate_features':int(X.duplicated().sum()), 'm_equals_250f':True,
        'dev_n':len(dev), 'dev_positive':int(y.iloc[dev].sum()), 'test_n':len(test),
        'test_positive':int(y.iloc[test].sum()), 'sha256':checksum, 'source':URL,
        'feature_overlap_dev_test':len(set(map(tuple,X.iloc[dev].values)) & set(map(tuple,X.iloc[test].values)))}
    save_json(out/'data_quality.json', quality)
    df.describe().to_csv(out/'descriptive.csv')
    cvrows, abrows, learnrows, params, searchrows, oof = [], [], [], [], [], []
    for fold, (a,b) in enumerate(folds):
        tr, va = dev[a], dev[b]
        inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED+100+fold).split(X.iloc[tr], y.iloc[tr]))
        fitted = {}
        for name in NAMES:
            search = fit_search(name, X.iloc[tr], y.iloc[tr], inner)
            fitted[name] = search.best_estimator_
            p = search.predict_proba(X.iloc[va])[:,1]
            cvrows.append({'fold':fold, 'repeat':fold//5, 'model':name, **metrics(y.iloc[va],p,va)})
            params.append({'fold':fold, 'model':name, 'params':search.best_params_})
            for i, par in enumerate(search.cv_results_['params']):
                searchrows.append({'fold':fold, 'model':name, 'params':json.dumps(par),
                    'inner_ap':search.cv_results_['mean_test_score'][i]})
            oof.extend({'fold':fold, 'id':int(j), 'y':int(y.iloc[j]), 'model':name, 'p':float(v)} for j,v in zip(va,p))
        base = fitted['svm']
        for variant, change in [('with_m',{'estimator__features__keep_m':True}),
                ('ratio',{'estimator__features__ratio':True}),
                ('linear',{'estimator__model__kernel':'linear'}),
                ('weight_toggle',{'estimator__model__class_weight':
                    None if base.estimator.named_steps['model'].class_weight == 'balanced' else 'balanced'})]:
            alt = clone(base).set_params(**change).fit(X.iloc[tr],y.iloc[tr])
            p = alt.predict_proba(X.iloc[va])[:,1]
            abrows.append({'fold':fold, 'variant':variant, **metrics(y.iloc[va],p,va)})
        # Тas pats išmokytas SVC: Platt ir nekalibruota sigmoidė, be papildomo mokymo.
        svcpipe = base.calibrated_classifiers_[0].estimator
        p = expit(svcpipe.decision_function(X.iloc[va]))
        abrows.append({'fold':fold,'variant':'raw_sigmoid',**metrics(y.iloc[va],p,va)})
        if fold < 5:
            for fraction in [0.25,0.5,0.75,1.0]:
                sub = tr if fraction==1 else train_test_split(tr,train_size=fraction,
                    stratify=y.iloc[tr],random_state=SEED+fold)[0]
                small = clone(base).fit(X.iloc[sub],y.iloc[sub])
                learnrows.append({'fold':fold,'fraction':fraction,'n':len(sub),
                    'train_ap':average_precision_score(y.iloc[sub],small.predict_proba(X.iloc[sub])[:,1]),
                    'ap':average_precision_score(y.iloc[va],small.predict_proba(X.iloc[va])[:,1])})
        pd.DataFrame(cvrows).to_csv(out/'cv_scores.csv',index=False)
        print(f'CV {fold+1}/25 baigta; {time.perf_counter()-started:.1f} s',flush=True)
    cv = pd.DataFrame(cvrows)
    summary = cv.groupby('model')[['ap','brier','p10','p20','p30']].agg(['mean','std'])
    summary.to_csv(out/'cv_summary.csv')
    pd.DataFrame(abrows).to_csv(out/'ablation.csv',index=False)
    pd.DataFrame(learnrows).to_csv(out/'learning.csv',index=False)
    pd.DataFrame(searchrows).to_csv(out/'inner_search.csv',index=False)
    pd.DataFrame(oof).to_csv(out/'oof_predictions.csv',index=False)
    save_json(out/'outer_params.json',params)
    paired = cv.pivot(index='fold',columns='model',values='ap')
    difference = paired.svm-paired.logistic
    # Nadeau-Bengio korekcija: foldai priklausomi, val/train = 1/4.
    se = np.sqrt((1/25+1/4)*difference.var(ddof=1))
    stat = difference.mean()/se if se else 0
    corrected_p = float(2*t.sf(abs(stat),24))
    interval = (difference.mean()+np.array([-1,1])*t.ppf(.975,24)*se).tolist()
    hypothesis = {'delta_ap':difference.mean(), 'corrected_t':stat, 'corrected_p':corrected_p,
        'corrected_ci95':interval, 'naive_t_p':ttest_rel(paired.svm,paired.logistic).pvalue,
        'naive_wilcoxon_p':wilcoxon(difference).pvalue,
        'h1_supported':bool(difference.mean()>=.03 and corrected_p<.05)}
    save_json(out/'hypothesis.json',hypothesis)
    # Plano atsarginis pasirinkimas: LR, jei iš anksto nustatyta H1 nepasitvirtina.
    chosen = 'svm' if hypothesis['h1_supported'] else 'logistic'
    final, finalparams = {}, {}
    innerdev = list(StratifiedKFold(3,shuffle=True,random_state=SEED+500).split(X.iloc[dev],y.iloc[dev]))
    for name in NAMES:
        search = fit_search(name,X.iloc[dev],y.iloc[dev],innerdev)
        final[name] = search.best_estimator_
        finalparams[name] = search.best_params_
        pd.DataFrame(search.cv_results_).to_csv(out/f'final_search_{name}.csv',index=False)
        joblib.dump(final[name],ROOT/'models'/f'{name}.joblib')
    save_json(out/'final_params.json',finalparams)
    # Atskirų vienodų požymių profilių atskyrimas; papildoma fiksuotų parametrų analizė.
    groups = pd.factorize(pd.MultiIndex.from_frame(X.iloc[dev]))[0]
    grows = []
    groupfolds = list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED+11).split(X.iloc[dev],y.iloc[dev],groups))
    save_json(out/'group_splits.json',[{'train':dev[a].tolist(),'validation':dev[b].tolist()} for a,b in groupfolds])
    for fold,(a,b) in enumerate(groupfolds):
        for name in NAMES:
            model = clone(final[name]).fit(X.iloc[dev[a]],y.iloc[dev[a]])
            grows.append({'fold':fold,'model':name,**metrics(y.iloc[dev[b]],model.predict_proba(X.iloc[dev[b]])[:,1],dev[b])})
    pd.DataFrame(grows).to_csv(out/'group_sensitivity.csv',index=False)
    # Testas naudojamas tik dabar, po parametrų ir naudojamo metodo užfiksavimo.
    predictions = {name:model.predict_proba(X.iloc[test])[:,1] for name,model in final.items()}
    pd.DataFrame([{'model':name,**metrics(y.iloc[test],p,test)} for name,p in predictions.items()]).to_csv(out/'test_scores.csv',index=False)
    save_json(out/'test_ci.json',bootstrap(y.iloc[test],predictions,test))
    pred = df.iloc[test].copy()
    pred.insert(0,'id',test)
    for name,p in predictions.items(): pred[name] = p
    pred.to_csv(out/'test_predictions.csv',index=False)
    stress = []
    for trial in range(30):
        rng = np.random.default_rng(SEED+1000+trial)
        missing = X.iloc[test].copy()
        missing = missing.mask(rng.random(missing.shape)<.2)
        noisy = X.iloc[test].copy().astype(float)
        # Triukšmas tik R/F/T; M atkuriamas iš F, kad nesulaužyti fizinio ryšio.
        for col in ['R','F','T']:
            noisy[col] = np.maximum(0,noisy[col]+rng.normal(0,.1*X.iloc[dev][col].std(),len(test)))
        noisy.M = 250*noisy.F
        for condition, altered in [('missing20',missing),('noise10',noisy)]:
            for name,model in final.items():
                p = model.predict_proba(altered)[:,1]
                stress.append({'trial':trial,'condition':condition,'model':name,**metrics(y.iloc[test],p,test)})
    pd.DataFrame(stress).to_csv(out/'robustness.csv',index=False)
    budget = math.ceil(len(test)*.2)
    order = rank_indices(predictions[chosen],test)
    selected = np.zeros(len(test),dtype=bool); selected[order[:budget]]=True
    pred['selected'] = selected
    pred['error_type'] = np.where(selected & (pred.y==0),'FP',np.where(~selected & (pred.y==1),'FN',''))
    pred[pred.error_type!=''].sort_values(chosen,ascending=False).to_csv(out/'errors.csv',index=False)
    pred.groupby(pd.cut(pred.R,[-1,5,15,np.inf],labels=['R<=5','5<R<=15','R>15']),observed=False).agg(
        n=('y','size'),positives=('y','sum'),invited=('selected','sum')).to_csv(out/'error_groups.csv')
    # Tiksli formulės patikra nepriklausomu RBF skaičiavimu.
    calibrated = final['svm'].calibrated_classifiers_[0]
    raw = calibrated.estimator.decision_function(X.iloc[test])
    reconstructed = predict_decision(calibrated.estimator,X.iloc[test])
    calibrator = calibrated.calibrators[0]
    pp = expit(-(calibrator.a_*raw+calibrator.b_))
    formula = {'max_decision_error':float(np.max(np.abs(raw-reconstructed))),
        'max_platt_error':float(np.max(np.abs(pp-predictions['svm']))),
        'A':calibrator.a_,'B':calibrator.b_,'support_vectors':len(calibrated.estimator.named_steps['model'].support_)}
    assert formula['max_decision_error']<1e-8 and formula['max_platt_error']<1e-8
    save_json(out/'formula_check.json',formula)
    confusion = {'TP':int((selected & (pred.y==1)).sum()),'FP':int((selected & (pred.y==0)).sum()),
        'FN':int((~selected & (pred.y==1)).sum()),'TN':int((~selected & (pred.y==0)).sum())}
    save_json(out/'run.json',{'seed':SEED,'selected_model':chosen,'budget':budget,
        'threshold':float(predictions[chosen][order[budget-1]]),'confusion':confusion,
        'expected_donations':float(predictions[chosen][order[:budget]].sum()),
        'seconds':time.perf_counter()-started,'python':platform.python_version(),
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__}})
    joblib.dump(final[chosen],ROOT/'models'/'selected.joblib')
    print(f'Baigta. Pasirinktas {chosen}; rezultatai: {out}',flush=True)


def rank_file(input_path, k, output_path):
    df = pd.read_csv(input_path)
    if not {'R','F','T'}.issubset(df.columns):
        raise ValueError('CSV turi turėti R, F, T stulpelius; donor_id neprivalomas')
    if len(df)==0 or not 1<=k<=len(df): raise ValueError('Netinkamas biudžetas')
    for col in ['R','F','T']:
        df[col] = pd.to_numeric(df[col],errors='raise')
        if (df[col].dropna()<0).any(): raise ValueError('Požymiai negali būti neigiami')
        if np.isinf(df[col]).any(): raise ValueError('Požymiai turi būti baigtiniai')
    if 'M' not in df: df['M']=250*df.F
    model = joblib.load(ROOT/'models'/'selected.joblib')
    p = model.predict_proba(df)[:,1]
    # Naujiems įrašams lygybę išsprendžia įvesties eilės numeris.
    order = np.lexsort((np.arange(len(df)),-p))
    result = df.copy(); result['probability']=p
    result['invite']=False; result.loc[result.index[order[:k]],'invite']=True
    result.iloc[order].to_csv(output_path,index=False)
    print(f'Atrinkta {k}; slenkstis {p[order[k-1]]:.4f}; rezultatas {output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('run')
    rank = sub.add_parser('rank'); rank.add_argument('--input',required=True)
    rank.add_argument('--budget',type=int,required=True); rank.add_argument('--output',default='kvietimai.csv')
    args = parser.parse_args()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter('always',ConvergenceWarning)
        if args.command=='run':
            run()
            save_json(ROOT/'results'/'warnings.json',{'count':len(captured),
                'messages':sorted(set(str(w.message) for w in captured))})
        else: rank_file(args.input,args.budget,args.output)


if __name__=='__main__':
    # Stabilus modulio vardas serializuotuose modeliuose ir paleidžiant CLI.
    from donation import main as stable_main
    stable_main()
