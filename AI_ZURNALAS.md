# AI naudojimo žurnalas

2026-09-18. Darbas pagal Tado Mikalajūno, PEPfm-26, pateiktą užduotį ir kolokviumo įgyvendinimo planą.

## Svarbiausios užklausos

1. Naudotojo užklausa: pagal jau parengtą planą įgyvendinti galutinio egzamino dalį. Priimta: dažnis, logistinė regresija, SVM RBF Platt ir MLP; vienodi skaidymai, AP, precision@k, Brier, abliacijos ir atsparumo bandymai.

## AI Nepatikrintos prielaidos

- CV dalys persidengia, todėl paprastas suporuotas t testas netinka interpretuoti jų kaip nepriklausomų imčių. Patikrinus oficialų statistinio palyginimo pavyzdį [5], pagrindinei išvadai pasirinkta Nadeau–Bengio korekcija. Nekoreguotos p reikšmės pateikiamos tik papildomai.
-LR tikimybės negali būti laikomos kalibruotomis vien dėl modelio pavadinimo. Testo Brier ir tikimybių suma parodė per aukštą balanced LR skalę. Tai ataskaitoje įvardinta kaip neigiamas rezultatas, o ne nutylėta. UCI rinkinio žymės, DOI, metodų bibliografiniai duomenys ir bibliotekos formulės patikrinti oficialiuose šaltiniuose.

## Šaltinių ir rezultatų patikra

- Oficialioje UCI rinkinio kortelėje patikrinti 748 įrašai, 4 požymiai, 2007 m. kovo žymė, miestas, DOI ir licencija. Pačiame duomenų faile patikrinta M=250F, klasės skaičiai ir SHA-256.
- Oficialiuose Springer ir Nature puslapiuose patikrinti SVM bei MLP metodų straipsnių bibliografiniai duomenys. Pilni mokami tekstai nebuvo perskaityti.
- AP apibrėžimas, Platt formulė, SVC ir koreguoto statistinio palyginimo pavyzdys patikrinti oficialioje scikit-learn dokumentacijoje. Tiesioginė Cornell PDF nuoroda buvo nepasiekiama, todėl jos turinys neteikiamas kaip tiesiogiai patikrintas.
- Paleistas visas eksperimentas su fiksuotomis sėklomis, išsaugoti skaidymai ir vidiniai kandidatų balai. Praėjo 5 patikros: galutinio testo izoliacija, grupinių profilių atskyrimas, išsaugotų modelių ir rezultatų sutapimas, RBF formulės lygybė ir tikslus k biudžetas. MLP konvergavimo įspėjimai išsaugomi, o ne nutylimi.
