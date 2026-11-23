# Third-Party Notices

PsychStrata CDSS is licensed under the GNU General Public License, version 3 or
later (see [LICENSE](LICENSE)). This file lists third-party components and data
sources that are redistributed with, or required at runtime by, this software.
It currently covers the components used by the pharmacogenomics (PGx) feature;
entries are added as dependencies are introduced.

## License compatibility

Every bundled dependency is GPLv3-compatible: the backend uses MIT, BSD-2/3-Clause,
Apache-2.0, PSF-2.0 and MPL-2.0 packages, and the frontend tree is MIT, ISC, BSD,
Apache-2.0, MPL-2.0, BlueOak-1.0.0, Unlicense, Zlib and 0BSD. Apache-2.0 is
compatible with GPLv3 but **not** with GPLv2, so do not downgrade the project
license to `GPL-2.0-only`. MPL-2.0 files are combined under GPLv3 via the
secondary-license clause, so keep their file-level notices intact.

---

## PyPGx 0.27.0 — MIT License

Used at runtime by the backend for star-allele validation, activity scores,
metaboliser phenotypes, and the bundled CPIC recommendation/guideline tables
(`pypgx.load_cpic_table()`, `pypgx.load_recommendation_table()`).

Upstream: <https://github.com/sbslee/pypgx>

```
MIT License

Copyright (c) 2020 Seung-been "Steven" Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

If PyPGx is used in a published analysis, its authors ask that the program
version be reported and the following article cited:

- Lee et al., 2022. *ClinPharmSeq: A targeted sequencing panel for clinical
  pharmacogenetics implementation*. PMID: 35821869.

### Beagle is not distributed

The PyPGx wheel also contains `pypgx/api/beagle.22Jul22.46e.jar`, which is
licensed under GPL-3.0-or-later. That license is compatible with this project's
license, so shipping it would not be a licensing conflict — but the distributor
would then have to supply Beagle's complete corresponding source and keep its
notices. This project does not use PyPGx phasing or imputation pipelines, so
`backend/Dockerfile` removes the JAR (and the `uv` cache that contains it) from
the image. Do not reintroduce the JAR without shipping Beagle's corresponding
source as well.

---

## `fuc` 0.38.0 — MIT License

Transitive dependency of PyPGx.

Upstream: <https://github.com/sbslee/fuc>

```
MIT License

Copyright (c) 2021 Seung-been "Steven" Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## CPIC — CC0 1.0 Universal

Source: Clinical Pharmacogenetics Implementation Consortium (CPIC®),
<https://www.clinpgx.org/page/cpicLicensing>.

The PGx feature reproduces CPIC drug recommendations verbatim from a dated
snapshot (`backend/app/defaults/cpic_recommendations.json`, retrieved from
`https://api.cpicpgx.org/v1/recommendation`) and displays CPIC gene/drug
levels, guideline links, and primary-literature PMIDs from the CPIC table
bundled with PyPGx.

CPIC states that its curated content is available free of restriction under
CC0 1.0 Universal (<https://creativecommons.org/publicdomain/zero/1.0/>).
Attribution is therefore not legally required, but CPIC requests it:

- CPIC®. <https://www.clinpgx.org/page/cpicLicensing> (accessed 2026-11-19).
- Cite the primary publication listed for each guideline; the PMIDs are shown
  in the user interface next to each recommendation.
- CPIC guidelines are revised as new evidence emerges. The guideline links in
  the interface point to the current versions; the snapshot date indicates when
  the reproduced text was retrieved.
- The CPIC logo and acronym may not be reproduced on another website or in
  advertising material without permission from NIH. The acronym is used here
  only to attribute the guideline content, not to imply endorsement, and no
  CPIC logo is included in this project.

The CPIC database and API are also subject to the CPIC Database and API User
Agreement. The snapshot records the API URL and the retrieval date; the CPIC
API does not expose a dataset version number, so the retrieval date identifies
the snapshot.

CPIC disclaimer (as published by CPIC): CPIC guidelines reflect expert
consensus based on clinical evidence and peer-reviewed literature available at
the time they are written and are intended only to assist clinicians in
decision making. New evidence may have emerged since a guideline was published
or since data were updated on the CPIC site. Users should consult the relevant
guideline before using CPIC data for any gene/drug pair.

---

## PharmGKB — CC BY-SA 4.0

Source: PharmGKB, <https://www.pharmgkb.org>, data usage policy
<https://www.clinpgx.org/page/dataUsagePolicy>.

PyPGx ships tables that incorporate PharmGKB-derived annotations (for example
the `PharmGKBLevel` column of its CPIC table). These tables are redistributed
**unmodified** as part of the PyPGx dependency. The PharmGKB-derived values are
not displayed by this application and are not merged into application code.

PharmGKB content is licensed under Creative Commons
Attribution-ShareAlike 4.0 International
(<https://creativecommons.org/licenses/by-sa/4.0/>). Redistribution therefore
requires attribution to PharmGKB, a link to the license, and an indication of
any changes; the tables are redistributed unmodified. They are separate data
files, not part of the GPL-licensed program: they remain under CC BY-SA 4.0
when redistributed, so do not copy them into program source files or relicense
them under the GPL.

---

## PharmVar — CC BY-SA 4.0

Source: Pharmacogene Variation Consortium (PharmVar),
<https://www.pharmvar.org/terms-and-conditions>.

Star-allele definitions and genomic positions in PyPGx's `allele-table.csv` and
`variant-table.csv` derive from PharmVar. These tables are redistributed
**unmodified** as part of the PyPGx dependency. The application itself does not
query PharmVar and stores only star-allele names, which it passes to PyPGx.

PharmVar content is licensed under Creative Commons
Attribution-ShareAlike 4.0 International
(<https://creativecommons.org/licenses/by-sa/4.0/>). PharmVar additionally
restricts use of its data to research purposes and prohibits offering the data
for sale as a commercial item. This project is a research/demonstration
dashboard and does not sell the data; deployments with commercial intent must
review the PharmVar terms independently. As with the PharmGKB tables, the
PharmVar-derived files are separate data files and stay under CC BY-SA 4.0 when
redistributed; they are not relicensed under the GPL.

---

## Not used

The PGx feature contains no content from the following sources. This is noted
because related licensing questions are often raised together with CPIC:

- **Dutch Pharmacogenetics Working Group (DPWG)** recommendations and rules.
  Reproducing DPWG guidance requires permission from DPWG.
- **safety-code.org / PREPARE trial safety cards.** The underlying software is
  licensed under AGPL-3.0. Neither the code, the card assets, nor the card
  format are used here.
