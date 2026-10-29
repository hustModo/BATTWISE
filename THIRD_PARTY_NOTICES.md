# Third-party notices

## EoH operator framework

`Evo_Textgrad_battery_eoh_deepseek/baselines/eoh/` contains an adapted EoH
implementation. The preserved source mapping identifies upstream revision
`d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71` of
[FeiLiu36/EoH](https://github.com/FeiLiu36/EoH).

The upstream [MIT license at that revision](https://raw.githubusercontent.com/FeiLiu36/EoH/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/LICENSE)
is included as `Evo_Textgrad_battery_eoh_deepseek/baselines/eoh/LICENSE`.
The upstream source mapping and bibliographic attribution are retained in
`baselines/eoh/readme.md`. These third-party authors and URLs must not be removed
to imply original authorship of their operator framework.

BATTWISE adds the battery workflow, agent collaboration, evaluation integration,
and reflective guidance. EoH is an attributed implementation component, not the
public name of BATTWISE or a newly claimed contribution.

## Experiment representations

KPCA uses scikit-learn. FC-VAE and DECM are repository implementations whose
source comments identify the methodological references and implementation
choices. Their inclusion does not establish independent author verification of
every historical setting. See `docs/EXPERIMENTS.md`.

## Runtime dependencies

Python and JavaScript dependencies are installed separately through package
managers. Their package licenses continue to apply. `node_modules`, Python
site-packages, browser binaries, and vendor build outputs are not redistributed
in this repository.

## FunSearch materials

`comparisons/funsearch/candidate_programs.txt` preserves candidate function text.
It is not the official FunSearch engine, an independent implementation of that
engine, or a complete FunSearch experiment runner. No third-party engine is
relicensed by this repository.
