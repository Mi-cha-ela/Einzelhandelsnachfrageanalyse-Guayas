# Einzelhandelsnachfrageanalyse-Guayas
In diesem Projekt wird die tägliche Nachfrage (unit_sales) für Einzelhandelsprodukte in der Region Guayas auf Basis des Kaggle-Datensatzes „Corporación Favorita“ prognostiziert. Der Fokus liegt auf Datenaufbereitung, Feature Engineering sowie dem Vergleich eines XGBoost-Baselinemodells mit einem LSTM-Ansatz.

## Datenhinweis

Die Originaldatensätze sind aufgrund ihrer Größe nicht im GitHub-Repository enthalten.
Bitte lade die Daten vom Kaggle-Wettbewerb „Favorita Grocery Sales Forecasting“ herunter:
https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting

Speichere die CSV-Dateien anschließend im Ordner /data, damit die Skripte korrekt ausgeführt werden können.

Einzelhandelsnachfrageanalyse
In diesem Projekt wird die tägliche Nachfrage (unit_sales) für Einzelhandelsprodukte in der Region Guayas auf Basis des Kaggle-Datensatzes „Corporación Favorite“ prognostiziert. Der Fokus liegt auf Datenaufbereitung, Feature Engineering sowie dem Vergleich eines XGBoost-Baselinemodells mit einem LSTM-Ansatz.

Nachfrageprognose für den Einzelhandel – Guayas (Woche 2)
Ziel
Prognose der täglichen Verkaufsmenge ( unit_sales) für Stores in der Region Guayas auf Basis vorbereiteter Zeitreihendaten (Q1 2014).

## Daten
Region: Guayas
Produktfamilien: LEBENSMITTEL I, GETRÄNKE, REINIGUNG
Zeitraum: 01.01.2014 – 31.03.2014
Zielvariable: Stückverkäufe
Modell
Modell: XGBoost-Regressor
Features: Kalendermerkmale, Lags, Rolling Mean, Store- & Item-Metadaten
Zug/Test: Jan–Feb / März (chronologisch)
Ergebnisse
MAE: XX.X
RMSE: XX,X
Kurzfristige Lags (lag_1) dominieren die Prognose
Kalender- und Store-Effekte unterstützt
Reflexion – Woche 2 (Guayas, XGBoost)
Diese Woche wurde auf Basis der in Woche 1 vorbereiteten Guayas-Daten (Top-3-Produktfamilien: LEBENSMITTEL I, GETRÄNKE, REINIGUNG) ein erstes Prognosemodell erstellt. Der Analysezeitraum wurde auf das erste Quartal 2014 (01.01.–31.03.) begrenzt und chronologisch in Trainingsdaten (Januar/Februar) und Testdaten (März) unterteilt, um Datenlecks zu vermeiden.

Das Feature Engineering umfasst Kalendermerkmale (Wochentag, Monat, Wochenende) sowie zeitliche Merkmale wie Lag-Features und Rolling Means. Das XGBoost-Modell stellte eine funktionierende Baseline dar, zeigte jedoch Grenzen aufgrund intermittierender Verkäufe (viele Nullen und seltene Peaks) sowie der starken Reduzierung des Datenausschnitts. In der Feature-Importance dominierten kurzfristige Signale (z. B. lag_1) und statische Artikel- bzw. Filialmerkmale, während Promotions und rollierende Mittelwerte in diesem Setup nur begrenzte Zusatznutzen lieferten.

## Fazit: Die Pipeline ist korrekt und reproduzierbar. Für realistischere Nachfrageprognosen auf Item-Ebene wären jedoch dichtere Zeitreihen (z. B. Auffüllen fehlender Tage) oder alternative Modellierungsebenen und -ansätze besser geeignet.

Zur zusätzlichen Analyse der zeitlichen Abhängigkeiten wurde ein Autokorrelationsdiagramm (ACF) der aggregierten Tagesumsätze für Guayas (Q1 2014) erstellt. Das Diagramm zeigte eine deutliche kurzfristige Autokorrelation (Lag 1) sowie Hinweise auf eine wöchentliche Struktur (Lag 7), was die Verwendung von Lag-Features grundsätzlich rechtfertigt. Gleichzeitig nahm die Autokorrelation schnell ab, was auf eine hohe Variabilität und unregelmäßige Nachfrage auf Item-Ebene hinwies. Dies erklärt, warum einfache Lag-Features im Modell vorherrschen, während komplexere rollierende Features nur begrenzte Zusatznutzenlieferten.

## Nächste Schritte / Verbesserungen
Zeitreihen verdichten: Für jede (store_nbr, item_nbr)-Kombination einen vollständigen Tagesindex erzeugen und fehlende Tage mit unit_sales=0 auffüllen, damit lag_7 und roll_mean_7 informativ werden.

Aggregation testen: Prognose auf Store×Family-Ebene (statt Store×Item) als stabilere Zwischenstufe vergleichen.

Funktionen erweitern: Feiertage, Ölpreis, Transaktionen und Promotion-Intensität (z. B. Promo-Anteil pro Woche) integrieren.

Zieltransformation: log1p(unit_sales) testen, um Peaks abzumildern.

## Modellvergleich: Alternativen wie HistGradientBoostingRegressor (sklearn), LightGBM (optional) oder LSTM/GRU (Bonus) ausprobieren und MAE/RMSE vergleichen.

Zur Validierung der zeitlichen Abhängigkeiten wurde ein ACF-Plot der differenzierten Tagesumsätze erstellt. Signifikante Autokorrelationen bei Lag 7 und Vielfachen bestätigen eine wöchentliche Saisonalität und rechtfertigen die Verwendung von Lag- und Rolling-Features im XGBoost-Modell.

## Optional: LSTM-Modell (konzeptioneller Vergleich)
Ein LSTM-Modell wurde als alternative Zeitreihenmethode in Betracht gezogen, da es sequenzielle Abhängigkeiten explizit modellieren kann und insbesondere bei dichten Zeitreihen Vorteile bietet. Für den vorliegenden Datenausschnitt (Guayas, Q1 2014) ist die Nachfrage auf Item-Ebene jedoch stark intermittierend, was eine Aggregation (z. B. Store×Family) für LSTM-Modelle erforderlich machen würde.

Eine praktische Implementierung wurde im Rahmen dieses Projekts nicht durchgeführt, da die verwendete Python-Version (3.14) aktuell nicht von TensorFlow unterstützt wird. Konzeptionell wäre ein LSTM insbesondere auf aggregierter Ebene (z. B. tägliche Verkäufe pro Produktfamilie) sinnvoll und könnte in zukünftiger Arbeit mit geeigneter Umgebung evaluiert werden.

Im Vergleich dazu eignet sich XGBoost besser als robuste Baseline für sparse tabellarische Zeitreihendaten mit vielen erklärenden Variablen.

# #„Alternativ wäre eine Implementierung mit PyTorch möglich gewesen, da dieses Framework neue Python-Versionen schneller unterstützt. Dies liegt jedoch außerhalb des Umfangs dieser Aufgabe.“

## In Colab Mini Modell erstellt:

## 📌 Ergebniszusammenfassung LSTM
Für die Region Guayas wurde zusätzlich ein LSTM-Modell auf aggregierter Ebene trainiert. Die täglichen Käufe wurden pro Produktfamilie zusammengefasst und für das erste Quartal 2014 modelliert. Für die Familie GROCERY I gab sich ein MAE von 340 und ein RMSE von 360. Die vergleichsweise hohen Fehlerwerte sind auf die Aggregation über alle Stores und Artikel zurückzuführen und liegen im realistischen Bereich der täglichen Gesamtverkäufe.

🧠 Kurze Einordnung / Vergleich
Im Vergleich zum XGBoost-Modell auf Store-Item-Ebene ist das LSTM-Modell weniger präzise, ​​da es auf einer deutlich größeren Aggregation und mit wenigen Eingangsmerkmalen trainiert wurde. Das Experiment zeigt jedoch, dass LSTM-Modelle grundsätzlich für aggregierte Nachfrageprognosen geeignet sind, bei kurzen Zeitreihen jedoch limitiert bleiben.

Reflexion – Woche 3
In Woche 3 wurde auf Basis der Q1-2014-Daten für Guayas ein XGBoost-Modell trainiert und mit MLflow systematisch evaluiert. Durch einen streng chronologischen Train-/Test-Split konnten realistische Prognosefehler berechnet werden.

Die Ergebnisse zeigen, dass XGBoost gegenüber der Baseline eine Verbesserung erzielt, insbesondere in MAE und RMSE. Gleichzeitig fällt der MAPE aufgrund stark intermittierender Nachfrage auf Item-Ebene sehr hoch aus, was die eingeschränkte Eignung prozentualer Fehlermaße in diesem Kontext verdeutlicht.

Die Analyse unterstreicht, dass Nachfrageprognosen auf aggregierter Ebene (z. B. Store × Familie) deutlich stabiler und besser modellierbar sind, während Item-Level-Forecasts spezielle Ansätze wie Croston erfordern. Insgesamt liefert das Projekt eine saubere, reproduzierbare Forecasting-Pipeline und ein realistisches Verständnis der Grenzen datengetriebener Nachfrageprognosen.

## 🗓️  – Streamlit-App
## Ziel
Businesstaugliche Oberfläche

Planer können:

Filiale wählen

SKU oder Familie auswählen

Zeitraum wählen

Wettervorhersage

CSV-Export

Warum Streamlit?

schnell entwickelt

wenig Code

Ideal für Data Science Prototypen

sofort visuell

## 🎯 Zentrale Erkenntnisse
Retail-Daten sind oft zeitweilig

Aggregation stabilisiert Prognosen

Boosting-Modelle schlagen Deep Learning bei kleinen/sparse Datensätzen

MLflow verbessert die Reproduzierbarkeit stark

Streamlit eignet sich ideal für schnelle Deployment-Demos

🧠 Technische Besonderheiten
Warum XGBoost?
Beste Performance auf tabellarischen Daten

Warum LSTM nur Colab?

TensorFlow unterstützt Python 3.14 noch nicht

Warum keine reinen Zeitreihenmodelle?

Zu viele parallele Serien + Lücken

Warum Aggregation sinnvoll?

Stabilere Signale

🎤 Kurzfazit
Es wurde eine reproduzierbare Demand-Forecasting-Pipeline mit XGBoost als robustem Baseline-Modell aufgebaut. Ein zusätzlicher LSTM-Vergleich wurde aufgrund von TensorFlow-Kompatibilität in Colab trainiert, jedoch auf sparse Retail-Daten geringerer Genauigkeit. Die Ergebnisse wurden über MLflow versioniert und in einer Streamlit-App interaktiv bereitgestellt.





Meine persönlichen Daten dürfen nicht weitergegeben werden.
