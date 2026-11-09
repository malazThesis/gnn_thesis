# Broad Real-World Screening — Validation-Selection-Fix

## Prüfung (9. November 2026)

Grundlage: die Originaldateien `gnn_thesis-main.zip` und `thesis_plan.zip`.

- Die bisherige Thesis-Tabelle hat 60 Einträge. **Alle 60** erreichen die
  maximale verfügbare mittlere Validation Accuracy ihrer jeweiligen
  Datensatz/Phase/Suchraum-Gruppe.
- Die ältere Implementierung sortierte bei der Modellwahl nach
  `mean_best_test_acc`, bevor sie `mean_best_val_acc` berücksichtigte.
  Wenn man sie erneut auf die 60 gespeicherten Gruppensummaries anwenden
  würde, käme man in 18 von 60 Fällen auf eine *andere Konfiguration*.
  Das bedeutet nicht, dass die 18 finalen Thesis-Zellen falsch sind:
  Die finale Tabelle wurde anhand der Validation-Werte rekonstruiert.
- Zwei Suchgruppen haben echte Gleichstände beim maximalen Validation-Wert:
  1. Coauthor-Physics, Phase 1 Baseline: GCN L2 H64 und GCN L4 H64
     (beide 93,80 % Validation);
  2. Cornell, Phase 2 Full: GraphSAGEPairNorm L8 H64 und L8 H128
     (beide ca. 60,84746 % Validation).
- Hier wird eine explizite testunabhängige Regel festgelegt:
  **höchste mittlere Validation Accuracy; bei Gleichstand weniger Layer,
  dann kleinere Hidden Dimension, dann Modellname alphabetisch**.
- Mit dieser Regel stimmt das Reanalyseergebnis in 59/60 Fällen mit
  der ursprünglichen Thesis-Tabelle überein.
  EIN sichtbarer Tabelleneintrag ändert sich:
  Coauthor-Physics, Phase 1 Baseline:
     ALT: GCN L4 H64, 92,83 ± 0,29 % Test
     NEU: GCN L2 H64, 93,20 ± 0,54 % Test.
  Das ist keine Veränderung der zugrunde liegenden Trainingsdaten.

## Inhalt

- `validation_selection.patch`: Git-Patch für 53 relevante
  Skripte (51 Aggregationsskripte + zwei globale Vergleichsskripte)
  mit insgesamt 76 korrigierten Sortierstellen.
- `rebuild_screening_validation.py`: reproduzierbare Auswahl aus 60
  bereits gespeicherten `*_grouped.csv`-Dateien, ohne Training.
- `screening_validation_selected.csv`: alle 60 neu ausgewählten Ergebnisse.
- `screening_changes_vs_thesis.csv`: die geänderte Tabellenzelle.
- `figures/fig06a_realworld_performance_depth.png`: zu dieser Auswahl passende Figure 6.
- `figures/fig06b_realworld_pairnorm_effects.png`: zu dieser Auswahl passende Figure 7;
  links Full minus Baseline bei validation-selektierten Konfigurationen,
  rechts unveränderte Mittelwerte gematchter Konfigurationen.

## Anwendung auf dem Cluster

SICHERUNG: Vorab Änderungen committen oder `git diff` sichern.

Das ZIP auf den Cluster übertragen, z. B. in das Projektverzeichnis,
anschließend in dem Projektverzeichnis:

    unzip validation_selection_fix.zip -d selection_fix
    git apply --check selection_fix/validation_selection.patch
    git apply selection_fix/validation_selection.patch
    ~/venvs/gnn-thesis-gpu/bin/python selection_fix/rebuild_screening_validation.py

Der letzte Befehl erzeugt:
  `runs/broad_screening_validation_selected_rebuilt.csv`

Es werden KEINE Trainingsläufe gestartet.
Die beiden Abbildungen liegen in `selection_fix/figures/` und
können die bisherigen Abbildungsdateien nach Kontrolle ersetzen.

WICHTIG: Patch und neue CSV-Zusammenfassung laufen nur
bei unverändert vorhandenen `runs/*_grouped.csv`-Dateien wie im Archiv.
Die Abbildungen werden nicht automatisch in dein LaTeX-Projekt kopiert.
Alte `best_grouped.csv`-Dateien, die vor dem Fix entstanden sind, sollten
nicht weiter für neue Auswertungen genutzt werden, bevor sie nach dem
korrigierten Skript neu erzeugt oder durch die neue Reanalyse ersetzt werden.

## Thesis-Anpassungen (bei Verwendung der oben genannten Tie-Break-Regel)

1) Coauthor-Physics, Phase 1 Baseline in Table 2 ersetzen:

    \makecell{GCN\\L2, H64\\$93.20\pm0.54$\%}

2) Im Fließtext des Coauthor-Physics-Absatzes:

    Coauthor-Physics retains comparatively stable performance,
    with selected baseline accuracy changing from
    $93.20\pm0.54\%$ in the shallow phase to
    $92.30\pm0.56\%$ in the deeper phase.
    The difference is approximately 0.90 percentage points.

3) Im Abschnitt Experimental Design nach der Modellwahl:

    Ties in mean validation accuracy were resolved by selecting
    the configuration with fewer layers, followed by the smaller
    hidden dimension and alphabetical model order.

4) Beide Abbildungen (Figure 6 und Figure 7) durch die beiliegenden
   Versionen ersetzen, da der Coauthor-Physics-Wert sonst nicht
   mehr zu Tabelle und Fließtext passt.

Die bisherige Erklärung getrennter Baseline/Full-Trainingsläufe
in der Table-2-Caption kann bestehen bleiben.
