# Published marker annotations

Source: Bilous, M., Buszta, D., Bac, J. et al. *Resolving sensitivity, specificity and signal contamination in Xenium spatial transcriptomics*. Nature Methods 23, 1152–1162 (2026). https://doi.org/10.1038/s41592-026-03089-8

Supplementary Tables 1–4 downloaded 2026-10-14:
https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41592-026-03089-8/MediaObjects/41592_2026_3089_MOESM4_ESM.xlsx

The workbook is preserved unchanged. The two CSVs were extracted with pandas.read_excel(sheet_name='ST 3' / 'ST 4', header=1), then to_csv(index=False). No gene annotations were changed. Names match the CSVs referenced by the authors' Bilous2026 notebook; their contents come from the published supplementary workbook, not the absent original CSVs.

The source article is CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
The Figure4 notebook adapts the analysis to Syndri's datasets, annotations, and correction methods. CosMx uses an intersection of the published lung panel marker sets with measured genes.
