# Kraujo donorystės tikimybės prognozė

Tadas Mikalajūnas, PEPfm-26. Intelektualiosios sistemos individualaus darbo programos kodas. Pagal UCI donorystės duomenis vertinamos aukojimo tikimybės ir sudaromas prioritetinis kvietimų sąrašas, kai žinomas kvietimų biudžetas. Programoje yra bendro dažnio ir logistinės regresijos atskaitos metodai, RBF SVM su Platt kalibravimu ir MLP.

## Paleidimas

Reikia Python 3.12 ar naujesnės versijos. Pirmą kartą duomenų atsisiuntimui reikia interneto. Windows PowerShell komandos šiame kataloge:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe donation.py run
```

Įdiegus priklausomybes, pagrindinį eksperimentą pakartoja viena komanda:

```powershell
python donation.py run
```

Komanda atsisiunčia [UCI Blood Transfusion Service Center](https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center) rinkinį, patikrina SHA-256, moko modelius ir įrašo eksperimento rezultatus į `results/`, o modelius į `models/`. Duomenys ir sugeneruoti failai į Git neįtraukiami. Pakartotinis vykdymas perrašo sugeneruotus modelių ir rezultatų failus. GPU nereikia; ankstesnis visas vykdymas šioje aplinkoje truko apie 105 sekundes.

Duomenų šaltinis: I-Cheng Yeh, 2008, DOI [10.24432/C5GS39](https://doi.org/10.24432/C5GS39), CC BY 4.0. Rinkinį sudaro 748 įrašai; tikslas `y=1` reiškia kraujo aukojimą 2007 m. kovą. Įvesties požymiai: `R` (mėnesiai nuo paskutinio aukojimo), `F` (aukojimų skaičius), `T` (mėnesiai nuo pirmojo aukojimo). `M=250×F`, todėl pagrindiniame modelyje pašalinamas.

## Vertinimo protokolas

- 20 % stratifikuotas galutinis testas atskiriamas prieš derinimą. Sėkla 2026.
- Likusioje dev dalyje naudojami tie patys 5×5 kartotiniai stratifikuoti išoriniai skaidymai visiems metodams. Kiekvienos išorinės mokymo dalies parametrus parenka vidinis 3 dalių CV pagal `average_precision`.
- Trūkstamų reikšmių pakeitimas mediana ir standartizavimas atliekami vamzdyno viduje. SVM kalibratorius apgaubia visą grandinę, todėl ir kalibravimo dalyse transformacijos išmokstamos tik iš mokymo įrašų.
- Rodikliai: average precision (ataskaitoje vadinamas PR-AUC), precision@10/20/30 % ir Brier balas. Išsaugoti CV, atskiro testo, bootstrap, abliacijos, atsparumo, grupinio jautrumo ir klaidų analizės rezultatai.
- H1: SVM vidutinis AP turi viršyti logistinę regresiją bent 0,03 ir koreguota p reikšmė turi būti mažesnė už 0,05. Kartotinio CV dalys priklausomos, todėl naudojamas Nadeau–Bengio koreguotas palyginimas. Jei H1 nepasitvirtina, pagal pradinį įgyvendinimo planą donorų atrankos pavyzdžiui pasirenkama logistinė regresija.

Šiame bandyme H1 nepasitvirtino: SVM−LR vidutinis AP skirtumas −0,018, koreguota p=0,431. Pagal plano atsarginę taisyklę `selected.joblib` yra LR, nors MLP pasiekė geriausius taškinius palyginimo rezultatus. LR tikimybės galutiniame teste buvo per aukštos; prieš praktinį naudojimą jas reikia patikrinti ir kalibruoti pagal naujos kampanijos duomenis. Istorinis aukojimo prognozavimas neįrodo kvietimo priežastinio poveikio.

## Donorų atranka

Po pagrindinio eksperimento galima apdoroti naują CSV:

```powershell
python donation.py rank --input pavyzdys.csv --budget 2 --output kvietimai.csv
```

CSV privalo turėti `R`, `F`, `T` stulpelius, o `donor_id` yra neprivalomas. Trūkstamas reikšmes pakeičia iš mokymo duomenų išmoktos medianos; neigiami ar begaliniai požymiai atmetami. Išvestyje pateikiama prognozuota `probability`, mažėjanti donorų eilė ir `invite` žyma tiksliai k donorų. Vienodų tikimybių atveju taikoma įvesties eilės tvarka. `pavyzdys.csv` yra dirbtinis funkcijos pavyzdys, ne eksperimento duomenys.

## Patikros ir failai

Po `run` vykdyti:

```powershell
python -m unittest discover -s tests -v
```

Penki testai tikrina testo izoliaciją, vienodų profilių grupių atskyrimą papildomame bandyme, išsaugotų modelių ir rezultatų sutapimą, nepriklausomą RBF formulės skaičiavimą bei tikslų top-k biudžetą. Testams reikia `run` sugeneruotų failų.

| Failas | Paskirtis |
|---|---|
| `donation.py` | Duomenų atsisiuntimas, kokybės patikra, eksperimentai ir donorų reitingavimo CLI |
| `requirements.txt` | Programos ir testų priklausomybės |
| `tests/test_results.py` | Eksperimento rezultatus tikrinantys testai |
| `pavyzdys.csv` | Nedidelė dirbtinė įvestis `rank` komandai |
| `AI_ZURNALAS.md` | Užduotyje reikalaujamas AI naudojimo ir klaidų patikros žurnalas |

`donation.py:predict_decision` atskirai skaičiuoja RBF sprendimo funkciją `g(z)=Σᵢaᵢexp(−γ‖sᵢ−z‖²)+b` ir palygina ją su `SVC.decision_function`. Platt tikimybė `1/(1+exp(A·g+B))` palyginama su `predict_proba`; didžiausios paklaidos įrašomos į `results/formula_check.json`.

Visos sėklos ir skaidymų indeksai fiksuoti programoje bei `results/splits.json`. Galutinis testas nenaudojamas hiperparametrams ar naudojimo taisyklei parinkti. Išbandyta scikit-learn 1.7.2 aplinka. MLP gali pasiekti 600 iteracijų ribą be konvergavimo; įspėjimai išsaugomi `results/warnings.json`.
