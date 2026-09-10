# aerial_classification_assets
het classiferen van objecten aan de hand van luchtfoto's

Deze software haalt de luchtfoto op bij PDOK, tekent daar vlakken op in voor groen en
verharding, en legt die naast de objectregistraties van de gemeente Amsterdam. Waar beeld
en registratie ver uit elkaar lopen komt een signaal. Amsterdam Centrum is het startpunt,
Noord staat erin als proeftuin voor de detectie zelf.

## Wat erin zit

De analyse werkt op vlakken en op vlakdekkendheid. Per geregistreerd vlak wordt bepaald
hoeveel ervan op de foto terug te zien is, en omgekeerd wordt gekeken welk maaiveld de
foto laat zien zonder dat daar een registratie tegenover staat. Puntobjecten zoals bomen
en lantaarnpalen zitten er bewust niet in, zie de kanttekening onderaan.

Twee registratiesets worden apart doorgerekend. De beheerregistratie is wat de gemeente
onderhoudt, dus groenobjecten en verhardingen uit de objectenregistratie openbare ruimte.
De BGT is de topografische tegenhanger, met begroeide terreindelen aan de groene kant en
wegdelen plus onbegroeide terreindelen aan de verharde kant. Die laatste twee horen in
één thema, want anders meldt elke rijbaan zich als ontbrekend terreindeel.

## Bronnen

De luchtfoto komt van de Landelijke Voorziening Beeldmateriaal via de WMTS van PDOK, laag
`Actueel_orthoHR` oftewel Luchtfoto Actueel Ortho 8cm RGB. De tegels worden opgehaald in
de RD-tegelmatrix (EPSG:28992), waarbij zoom 15 uitkomt op 10,5 cm per pixel en zoom 16 op
5,25 cm. Alles wordt lokaal gecachet, dus een tweede run over hetzelfde gebied kost geen
nieuwe downloads.

De registraties komen van `api.data.amsterdam.nl`, zonder sleutel of registratie. De
software vraagt ze op in RD en knipt ze op het analysegebied. Welke bronnen dat zijn zie
je met `python -m luchtfoto_objecten lagen`.

## De bladstand, en waarom dat hier uitmaakt

De 8cm-ortho wordt in het vroege voorjaar gevlogen, met kale bomen, zodat het maaiveld
zichtbaar is. Dat is precies goed voor verharding, maar vegetatie is er niet uit te halen.
Gemeten op de geregistreerde vlakken in de Plantage: de excess green index heeft een
mediaan van -0,007 binnen groenobjecten en -0,009 binnen verhardingen. Winters gras en
bestrating zijn op die opname dus niet te onderscheiden. Ook de textuur helpt niet, die
ligt op 0,03 voor allebei.

Daarom kiest de software zelf een tweede opname voor het groen. Hij bemonstert een klein
vlak uit het midden van het gebied op de beschikbare jaargangen, van nieuw naar oud, en
neemt de eerste die genoeg blad laat zien. Voor Centrum levert dat dit beeld op:

| laag | groene pixels |
|---|---|
| Actueel_orthoHR | 0,3% |
| 2026_orthoHR | 4,8% |
| 2026_quickorthoHR | 5,2% |
| 2025_orthoHR | 0,3% |
| 2025_ortho25 | 79,2% |

De verharding blijft dus van de nieuwste 8cm-foto komen, het groen van de nieuwste opname
met blad. Dat verschil in opnamemoment is zelf ook een bron van afwijkingen, houd dat in
gedachten bij het lezen van de signalen. Wil je het uitzetten, zet dan `groenlaag` in
`config/parameters.yaml` op `gelijk`, of vul een vaste laagnaam in.

## Installeren

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

In PyCharm wijs je `.venv` aan als interpreter, daarna draaien de scripts in `scripts/`
direct. De optionele SAM-module staat in `requirements-sam.txt` en is alleen nodig als je
contouren wilt laten aanscherpen door Segment Anything.

## Gebruiken

```bash
python -m luchtfoto_objecten gebieden
python -m luchtfoto_objecten analyse --gebied centrum_plantage
python -m luchtfoto_objecten analyse --bbox 122100 486450 122500 486800 --zoom 16
python -m luchtfoto_objecten registratie --gebied noord_ndsm
```

De gebieden staan in `config/gebieden.yaml`, in RD-coördinaten. Begin klein. Een vlak van
400 bij 350 meter is op zoom 15 ongeveer 250 tegels en draait in een halve minuut. Heel
stadsdeel Centrum staat er ook in, maar dat zijn ruim achtduizend tegels.

## Wat eruit komt

Per gebied verschijnt in `data/uitvoer/<gebied>/` een GeoPackage met alle lagen, dezelfde
lagen los als GeoJSON, de luchtfoto als GeoTIFF in RD, en een samenvatting in JSON en CSV.

De laag `signaleringen` is waar het om draait. Elk vlak heeft een status:

- `bevestigd`, de registratie komt overeen met het beeld
- `afwijkende_geometrie`, een deel van het vlak is terug te zien, de rest niet
- `niet_zichtbaar_op_luchtfoto`, op die plek toont de foto iets anders
- `ontbreekt_in_registratie`, de foto laat iets zien dat nergens geregistreerd staat
- `geregistreerd_als_andere_klasse`, wel geregistreerd, maar in een ander thema

Die laatste status voorkomt veel ruis. Een berm staat in Amsterdam als verharding
geregistreerd, ruim tachtig stuks alleen al in de Plantage, en die is op de zomerfoto
gewoon groen. Zonder dit onderscheid zou elke berm zich als ontbrekend groen melden.

De laag `witte_vlekken` beantwoordt de vlakdekkendheidsvraag: welk maaiveld valt in geen
enkel geregistreerd vlak. Randstroken smaller dan 75 centimeter worden weggefilterd, want
twee vlakdekkende bestanden over hetzelfde gebied leveren langs gedeelde randen altijd
flinterdunne verschillen op die nergens over gaan. In `signalering_raster` staan de
signalen geteld per cel van honderd meter, met een vlag zodra een cel boven de drempel
komt.

## Het analysegebied

Uitspraken worden alleen gedaan over maaiveld waar de gemeente ook echt iets registreert.
Panden en water gaan er via de BGT uit, en daarna wordt het gebied begrensd tot de
beheerkaart, de BGT-vlakken op percelen die Amsterdam in eigendom heeft. Dat scheelt
enorm. Zonder die begrenzing meldden binnentuinen en particuliere erven zich als
ontbrekend groen en kwam de dekkingsgraad van de beheerregistratie op 68 procent uit, met
de begrenzing op 90 procent. Uitzetten kan met `beperk_tot_beheergebied` in
`config/parameters.yaml`.

## Drempels

Alle drempels staan in `config/parameters.yaml`. De verhardingsdrempels zijn geijkt op de
geregistreerde verhardingsvlakken in Centrum: verzadiging heeft daar een mediaan van 0,08
met een p90 rond 0,40 door de rode klinkers, en textuur een mediaan van 0,03 met een p90
rond 0,10. Wie de drempels wil bijstellen voor een ander stadsdeel doet er goed aan die
meting daar opnieuw te doen, want Noord ziet er anders uit dan de binnenstad.

## Beperkingen

Onder dichte boomkronen is het maaiveld op een nadirfoto niet te zien. Verharding onder
een bomenrij telt daardoor als afwijking terwijl de registratie klopt. Hetzelfde geldt
voor geparkeerde voertuigen, bouwplaatsen en marktkramen, die verlagen de dekkingsgraad
zonder dat er iets mis is. Objecten aan de rand van het gebied zijn afgeknipt op de bbox,
hun oppervlakte valt dus lager uit.

Lantaarnpalen zaten in een eerdere opzet wel in de software, via detectie van hun
slagschaduw. Dat is er weer uit. Op de 8cm-ortho zijn de masten zelf nauwelijks te zien,
alleen het oplichtende armatuur en een dunne schaduwstreep, en in de binnenstad lopen die
door tussen schaduwen van gevels, hekken, bomen en auto's. Twee aanpakken zijn getest
tegen de geregistreerde lichtmasten in de Plantage en op Kattenburg, één die uitgaat van
dunne donkere lijnen en één die begint bij het heldere armatuur en vandaar de schaduw
volgt. Beide bleven op zinnige aantallen steken onder een precisie van 0,2 bij een recall
onder 0,2. Dat levert meer opruimwerk op dan het oplost. Voor puntobjecten zijn schuine
opnamen of een puntenwolk een realistischer route.
