CARBON FUNCTIONAL MATERIALS DATABASE (cfms_db) - Version 1.0
===============================================================================

The `cfms_db_1.csv` file is in version 1.0 and it was obtained from the
processsing of 10,975 articles which contained the terms **"biochar"** or
**"hydrochar"** either in the title, abstract or keywords. The title of each
article indexed in the `cfms_db.csv` is detailed in the file
`filename_id_1.csv`.

The carbon functional materials database (cfms_db) was generated from the
processing of scientific articles by the [a.RIX
engine](https://github.com/amaurijp/aRIX). Details on the use of the a.RIX
engine is presented in:

> *An automated domain-independent text reading, interpreting and extracting
> approach for reviewing the scientific literature.*
> Amauri J. Paula.
> Solid-Biological Interface Group (SolBIN), Department of Physics,
> Universidade Federal
> do Ceará – UFC, P.O. Box 6030, Fortaleza, CE, 60455-900, Brazil
>
> [https://arxiv.org/abs/2107.14638](https://arxiv.org/abs/2107.14638)

A complete first analysis of the cfms_db database is presented in the article:

> *Machine learning and natural language processing enable a data-oriented
> experimental design approach for producing biochar and hydrochar from
> biomass.* Chemistry of Materials. DOI: 10.1021/acs.chemmater.1c02961
>
> Amauri J. Paula,<sup>1</sup>
> Odair P. Ferreira,<sup>2</sup>
> Antonio G. Souza Filho,<sup>3</sup>
> Francisco Nepomuceno Filho,<sup>3</sup>
> Carlos E. Andrade,<sup>4</sup>
> Andreia F. Faria.<sup>5</sup>
>
> <sup>1</sup>Solid-Biological Interface Group (SolBIN), Department of Physics,
> Universidade Federal do Ceará – UFC, P.O. Box 6030, Fortaleza, CE, 60455-900,
> Brazil.
>
> <sup>2</sup>Laboratório de Materiais Funcionais Avançados (LAMFA),
> Department of Physics, Universidade Federal do Ceará – UFC, Fortaleza, CE,
> 60455-900, Brazil.
>
> <sup>3</sup>Department of Physics, Universidade Federal do Ceará – UFC,
> Fortaleza, CE, 60455-900, Brazil.
>
> <sup>4</sup>AT&T Labs Research, 200 South Laurel Avenue, Middletown, NJ,
> 07748, USA.
>
> <sup>5</sup>Engineering School of Sustainable Infrastructure & Environment,
> Department of Environmental Engineering Sciences, University of Florida,
> Gainesville, FL, 32611-6540, USA.

-------------------------------------------------------------------------------

Published on Chemistry of Materials (to appear)
-------------------------------------------------------------------------------

The features (columns in file `cfms_db.csv`) extracted are:

1. carbonization (synthesis) method / Physical Unit = None (categorical terms has no PU);
2. biomass precursor / Physical Unit = None (categorical terms has no PU);
3. carbonization temperature / Physical Unit = temperature;
4. carbonization time / Physical Unit = time;
5. surface area / PU = area weight<sup>-1</sup>;
6. particle size / Physical Unit = Euclidean distance;
7. adsorption capacity / Physical Unit = weight weight<sup>-1</sup>;
8. H/C weight ratios / Physical Unit = dimensionless;
9. C/N weight ratios / Physical Unit = dimensionless;
10. O/C weight ratios / Physical Unit = dimensionless;
11. C/O weight ratios / Physical Unit = dimensionless;
12. N/C weight ratios / Physical Unit = dimensionless;
13. C/H weight ratios / Physical Unit = dimensionless;
14. high heating value (HHV) / Physical Unit = energy weight<sup>-1</sup>.
