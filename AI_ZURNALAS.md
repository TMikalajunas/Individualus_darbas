# AI naudojimo žurnalas

2026-09-18. Darbas pagal Tado Mikalajūno, PEPfm-26, pateiktą užduotį ir kolokviumo įgyvendinimo planą. Žurnale užfiksuotas programos rengimo ir patikros procesas; jis nepriskiria studentui neatliktų rankinių veiksmų.

## Svarbiausios užklausos ir sprendimai

1. Naudotojo užklausa: pagal jau parengtą planą įgyvendinti galutinio egzamino dalį. Priimta: dažnis, logistinė regresija, SVM RBF Platt ir MLP; vienodi skaidymai, AP, precision@k, Brier, abliacijos ir atsparumo bandymai.
2. Darbo klausimas: kaip išvengti hiperparametrų parinkimo ir vertinimo duomenų sutapimo? Priimta: 20 % galutinis testas ir įdėtinis 5×5 išorinis CV su 3 dalių vidiniu CV. Atmesta prielaida, kad derinimo CV balas yra nepriklausomas įvertis.
3. Darbo klausimas: kaip kalibruoti SVM nesukeliant požymių paruošimo nutekėjimo? Priimtas `CalibratedClassifierCV` aplink visą imputacijos, standartizavimo ir SVC grandinę. Atmesta idėja standartizuoti visą dev rinkinį prieš CV.
4. Darbo klausimas: kaip patikrinti plano SVM formulę? RBF suma apskaičiuota nepriklausomai iš išmokytų atraminių vektorių ir palyginta su `decision_function`; Platt formulė palyginta su `predict_proba`.
5. Darbo klausimas: kaip patikrinti H1 persidengiančiose CV dalyse? Pagrindinei išvadai priimtas Nadeau–Bengio koreguotas testas; nekoreguoti t ir Wilcoxon pateikiami papildomai. H1 nepasitvirtino, pritaikyta plano LR atsarginė taisyklė. Atmestas galutinio testo MLP balu grindžiamas naudojamo modelio pakeitimas.

## Aptiktos AI klaidos ir nepatikrintos prielaidos

| Klaida arba prielaida | Aptikimas | Pataisymas ir patikra |
|---|---|---|
| F/T santykiui iš pradžių naudota `X.T`, tačiau pandas taip transponuoja lentelę | Pirmas tikras paleidimas pateikė `ValueError` dėl masyvų matmenų | Pakeista į `X['T']`; pakartotas visas eksperimentas ir F/T abliacija |
| Duomenų profilių sankirtos išraiškoje praleistas uždaromasis skliaustas | Python pateikė `SyntaxError` prieš modelių mokymą | Skliaustas pataisytas; programa paleista iki pabaigos |
| Paprastas suporuotas t testas gali pernelyg optimistiškai vertinti persidengiančių CV dalių skirtumą | Patikrinta oficiali scikit-learn statistinio palyginimo dokumentacija | Pagrindinei H1 išvadai naudota koreguota paklaida, o nekoreguoti testai laikomi papildomais |
| Logistinė regresija nebūtinai gerai kalibruota, ypač su balanced klasės svoriais | Testo Brier ir atrinktųjų tikimybių suma parodė per aukštą skalę | Rezultatas įvardytas kaip apribojimas; realiam planavimui siūloma naujų kampanijų kalibravimo patikra |

## Šaltinių ir rezultatų patikra

- Oficialioje UCI rinkinio kortelėje patikrinti 748 įrašai, 4 požymiai, 2007 m. kovo žymė, miestas, DOI ir licencija. Pačiame duomenų faile patikrinta M=250F, klasės skaičiai ir SHA-256.
- Oficialiuose Springer ir Nature puslapiuose patikrinti SVM bei MLP metodų straipsnių bibliografiniai duomenys. Pilni mokami tekstai nebuvo perskaityti.
- AP apibrėžimas, Platt formulė, SVC ir koreguoto statistinio palyginimo pavyzdys patikrinti oficialioje scikit-learn dokumentacijoje. Tiesioginė Cornell PDF nuoroda buvo nepasiekiama, todėl jos turinys neteikiamas kaip tiesiogiai patikrintas.
- Paleistas visas eksperimentas su fiksuotomis sėklomis, išsaugoti skaidymai ir vidiniai kandidatų balai. Praėjo 5 patikros: galutinio testo izoliacija, grupinių profilių atskyrimas, išsaugotų modelių ir rezultatų sutapimas, RBF formulės lygybė ir tikslus k biudžetas. MLP konvergavimo įspėjimai išsaugomi, o ne nutylimi.
