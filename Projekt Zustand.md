1. Technischer Start

Kurze Einzeltests prüften Datenladen, Training und Auswertung.

Kleine Cluster-Jobs prüften die Ausführung auf GPUs.

Kleine Job-Arrays prüften mehrere Konfigurationen gleichzeitig.

2. Erste synthetische CSBM-Tests

CSBM erlaubte die gezielte Kontrolle von Tiefe, Homophilie und Feature-Signal.

Flache Baselines und PairNorm-Modelle liefen meistens stabil.

Tiefe Baselines reagierten deutlich empfindlicher.

Schwierige Einstellungen zeigten die Unterschiede besonders klar.

Niedrige Homophilie und mittleres Feature-Signal dienten als Stresstest.

Die achtlagige Baseline wurde in dieser Einstellung häufig instabil.
PairNorm blieb deutlich robuster.

3. Erste Oversmoothing-Analyse auf CSBM

Tiefe Baselines zeigten sehr kleine Repräsentationsdistanzen.

Die graphbezogene Variation nahm stark ab.

Der effektive Rang fiel auf sehr niedrige Werte.

PairNorm hielt die Repräsentationen stärker auseinander.

PairNorm erhielt mehr graphbezogene Variation.

PairNorm erhielt einen deutlich höheren effektiven Rang.

Diese Resultate bestätigten den erwarteten Oversmoothing-Effekt.

4. Robustheit über mehrere Seeds

25 Kombinationen aus Graph und Initialisierungsseeds prüften die Stabilität.

Die Baseline war bei hoher Lernrate nur in 4 von 25 Läufen erfolgreich.

PairNorm war in 25 von 25 Läufen erfolgreich.

Der Unterschied hing damit nicht nur von einzelnen Seeds ab.

5. Lernratentests auf CSBM

Ein Lernratensweep untersuchte die Empfindlichkeit der Baseline.

Kleine bis mittlere Lernraten liefen deutlich stabiler.

Die beste robuste Baseline-Lernrate lag bei 0,005.

Ein breiter PairNorm-Sweep prüfte einen größeren Lernratenbereich.

PairNorm blieb über fast den gesamten Bereich stabil.

Am besten Checkpoint waren alle 250 PairNorm-Läufe erfolgreich.

Die schnellste robuste PairNorm-Einstellung konvergierte ungefähr 7,5-mal schneller als die beste robuste Baseline.

6. PairNorm-Komponenten

Separate Tests untersuchten Zentrierung und Skalierung.

Zentrierung allein hielt die Genauigkeit häufig hoch.

Zentrierung allein führte bei langen Läufen zu sehr großen Normen.

Skalierung allein kontrollierte die Norm besser.

Skalierung allein verhinderte bei sehr hoher Lernrate nicht immer den Kollaps.

Die vollständige Kombination blieb am stabilsten.

Ein Langzeittest über 1000 Epochen bestätigte die numerische Stabilität von PairNorm.

7. Real-World-Tests

Cora

Flache und tiefe Modelle liefen mit und ohne PairNorm.

Größere Tiefe verschlechterte die beste Leistung.

PairNorm brachte keinen einheitlichen Vorteil für alle Modelle.

CiteSeer

Flache und tiefe Varianten zeigten erneut einen Tiefenverlust.

PairNorm wirkte abhängig von Architektur und Konfiguration.

PubMed

Tiefere Modelle erreichten keine bessere Leistung.

PairNorm zeigte keinen universellen Vorteil.

Roman-Empire

Baselines und PairNorm-Modelle reagierten stark unterschiedlich.

Der PairNorm-Effekt hing deutlich von der Architektur ab.

Amazon-Photo

Mehrere Architekturen, Tiefen und Hidden-Dimensionen kamen zum Einsatz.

Tiefe Modelle blieben schwächer als flache Modelle.

PairNorm half nur in bestimmten Konfigurationen.

Amazon-Computers

Tiefe Modelle verloren erneut an Leistung.

PairNorm zeigte je nach Architektur unterschiedliche Effekte.

Coauthor-CS

Größere Tiefe verschlechterte die beste Konfiguration.

PairNorm brachte keinen durchgehend positiven Effekt.

Coauthor-Physics

Tiefe Varianten blieben schwächer.

PairNorm half nicht bei allen Architekturen gleich stark.

Actor

Die Resultate schwankten stark zwischen Architektur und Tiefe.

PairNorm zeigte keinen einheitlichen Vorteil.

Chameleon

Tiefe Modelle waren schwieriger zu optimieren.

PairNorm half nur in einzelnen Konfigurationen.

Squirrel

Baseline und PairNorm zeigten heterogene Unterschiede.

Größere Tiefe verschlechterte auch hier die Leistung.

Cornell

Die kleine Datensatzgröße führte zu stärkeren Schwankungen.

Tiefe Modelle blieben insgesamt schwächer.

Texas

Die Resultate hingen stark von den Splits ab.

PairNorm brachte keinen stabilen Vorteil über alle Konfigurationen.

Wisconsin

Die Unterschiede zwischen Baseline und PairNorm blieben uneinheitlich.

Größere Tiefe verschlechterte die Leistung.

Amazon-Ratings

Amazon-Ratings zeigte den stärksten und stabilsten PairNorm-Effekt.

Der Datensatz eignete sich deshalb für die detaillierte Hauptanalyse.

8. Gesamtbild der Real-World-Tests

Größere Tiefe verschlechterte die beste Leistung auf allen 15 Datensätzen.

PairNorm half nicht auf jedem Datensatz gleich stark.

Die stärksten Vorteile traten meistens bei GraphSAGE auf.

Amazon-Ratings lieferte den klarsten Real-World-Fall.

9. Erster Amazon-Ratings-Vergleich

PairNorm verbesserte GraphSAGE bei vier Schichten um ungefähr 4,23 Prozentpunkte.

Bei acht Schichten lag der erste Unterschied bei ungefähr 11,20 Prozentpunkten.

Die ersten Werte enthielten noch mögliche Lernrateneffekte.

Ein fairer Lernratensweep folgte deshalb als nächster Schritt.

10. Fairer Lernratensweep mit 400 Epochen

Sechs Lernraten und zehn identische Splits deckten jede Modell-Tiefen-Kombination ab.

Die Auswahl orientierte sich nur an der Validation-Accuracy.

PairNorm gewann bei vier Schichten ungefähr 2,90 Prozentpunkte.

PairNorm gewann bei acht Schichten ungefähr 4,00 Prozentpunkte.

PairNorm war bei beiden Tiefen auf allen zehn Splits besser.

Auch bei identischen Lernraten blieb PairNorm in allen getesteten Fällen überlegen.

11. Verlängerung auf 800 Epochen

Ausgewählte Konfigurationen trainierten bis 800 Epochen.

Die Baseline holte teilweise auf.

PairNorm blieb trotzdem besser.

Der Vorteil lag bei vier Schichten bei ungefähr 1,75 Prozentpunkten.

Der Vorteil lag bei acht Schichten bei ungefähr 3,46 Prozentpunkten.

12. Vollständiger Lernratensweep mit 800 Epochen

Insgesamt deckten 240 vollständige Läufe sechs Lernraten und zehn Splits ab.

Bei vier Schichten nutzten beide Modelle die beste Lernrate 0,003.

PairNorm gewann dort ungefähr 1,45 Prozentpunkte.

Bei acht Schichten nutzte GraphSAGE 0,003 und PairNorm 0,005.

PairNorm gewann dort ungefähr 3,33 Prozentpunkte.

PairNorm war bei beiden Tiefen auf allen zehn Splits besser.

Die gepaarten Unterschiede blieben statistisch signifikant.

Die achtlagige Baseline kollabierte bei mehreren höheren Lernraten.

PairNorm blieb bei denselben Lernraten stabil.

13. Tiefeneffekt auf Amazon-Ratings

GraphSAGE verlor von vier auf acht Schichten ungefähr 2,73 Prozentpunkte.

PairNorm verlor nur ungefähr 0,86 Prozentpunkte.

PairNorm reduzierte den Tiefenverlust um ungefähr 68,6 %.

Die Tiefeninteraktion war auf 9 von 10 Splits positiv.

14. Tiefentest bei identischer Lernrate

Alle vier Modell-Tiefen-Kombinationen nutzten die Lernrate 0,003.

GraphSAGE verlor erneut ungefähr 2,73 Prozentpunkte.

PairNorm verlor nur ungefähr 0,93 Prozentpunkte.

PairNorm reduzierte den Tiefenverlust um ungefähr 66,1 %.

Der Effekt war auf allen zehn Splits positiv.

Dieser Vergleich liefert den stärksten kontrollierten Nachweis für die geringere Tiefenempfindlichkeit.

15. Oversmoothing-Analyse auf Amazon-Ratings

Die Analyse verglich die letzte 128-dimensionale Hidden-Repräsentation.

PairNorm erhöhte bei acht Schichten die normalisierte Dirichlet-Energie von ungefähr 0,45 auf 1,04.

PairNorm erhöhte den effektiven Rang von ungefähr 3,25 auf 11,78.

Der effektive Ranganteil lag mit PairNorm mehr als dreimal so hoch.

Alle direkten Unterschiede zeigten auf allen zehn Splits dieselbe Richtung.

Die tiefen PairNorm-Repräsentationen blieben strukturreicher und weniger komprimiert.

16. Mechanistische Tiefeninteraktion

Die normalisierte Dirichlet-Energie sank bei GraphSAGE mit größerer Tiefe.

Mit PairNorm stieg die normalisierte Dirichlet-Energie.

Die Difference-in-Differences war auf allen zehn Splits positiv.

Der effektive Rang von PairNorm nahm mit Tiefe ab.

Trotzdem blieb der effektive Rang bei acht Schichten mehr als dreimal so hoch wie bei der Baseline.

PairNorm verhinderte den Rangverlust nicht vollständig, hielt aber ein deutlich höheres Niveau.

17. Zusammenhang zwischen Metriken und Accuracy

Splitweise Korrelationen verglichen Repräsentationsgewinne und Accuracy-Gewinne.

Keine Korrelation erreichte statistische Signifikanz.

Die Metriken trennen Baseline und PairNorm klar.

Die Metriken erklären bei zehn Splits nicht zuverlässig die Höhe des Accuracy-Gewinns.

Die mechanistischen Resultate gelten deshalb als starke Assoziation, nicht als kausaler Beweis.

18. Kontrolle nahe dem Validation-Optimum

Die Metriken am letzten Messpunkt vor dem Validation-Optimum bestätigten dieselben Unterschiede.

PairNorm lag bei allen vier Hauptmetriken und beiden Tiefen höher.

Alle acht Vergleiche zeigten auf allen zehn Splits dieselbe Richtung.

Die Resultate hingen damit nicht nur von Epoche 800 ab.

19. Kontrolle bei festen Epochen

Feste Epochen 400, 600 und 800 ermöglichten direkte zeitliche Vergleiche.

PairNorm lag bei beiden Tiefen und allen vier Hauptmetriken höher.

Bei acht Schichten lag die normalisierte Dirichlet-Energie nach 400 Epochen ungefähr zehnmal so hoch.

Nach 600 Epochen lag sie ungefähr 3,7-mal so hoch.

Nach 800 Epochen lag sie ungefähr 2,3-mal so hoch.

Der effektive Ranganteil blieb zu allen drei Zeitpunkten mehr als dreimal so hoch.

Alle 24 Vergleiche zeigten auf allen zehn Splits einen Vorteil für PairNorm.


20. Aktueller Hauptbefund

PairNorm reduziert den Tiefenverlust von GraphSAGE auf Amazon-Ratings um ungefähr zwei Drittel.

Der Vorteil bleibt nach fairer Lernratenauswahl bestehen.

Der Vorteil bleibt nach längerem Training bestehen.

Der Vorteil bleibt bei identischer Lernrate bestehen.

PairNorm erweitert zusätzlich den stabilen Lernratenbereich.

Tiefe PairNorm-Repräsentationen besitzen eine höhere graphbezogene Variation.

Tiefe PairNorm-Repräsentationen besitzen einen deutlich höheren effektiven Rang.

