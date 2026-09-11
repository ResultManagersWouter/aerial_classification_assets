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
neemt de eerste die genoeg blad laat zien. Dat levert bijvoorbeeld dit beeld op:

| laag | groene pixels |
|---|---|
| Actueel_orthoHR | 0,3% |
| 2026_orthoHR | 4,8% |
| 2026_quickorthoHR | 5,2% |
| 2026_quickortho25 | 0,0% |
| 2025_orthoHR | 0,3% |
| 2025_ortho25 | 79,2% |

Gemeten op `centrum_plantage`. Welke jaargangen er zijn verandert als PDOK er een
toevoegt, dus draai het zelf na voor je gebied; de meting staat in het logboek van elke
analyse en in `bladstand_gemeten` in de samenvatting.

De verharding blijft dus van de nieuwste 8cm-foto komen, het groen van de nieuwste opname
met blad. Dat verschil in opnamemoment is zelf ook een bron van afwijkingen, houd dat in
gedachten bij het lezen van de signalen. Wil je het uitzetten, zet dan `groenlaag` in
`config/parameters.yaml` op `gelijk`, of vul een vaste laagnaam in.

## Beginnen als je hier nieuw bent

Je hebt Python en een internetverbinding nodig, verder niets. Ontwikkeld op 3.12, en 3.10
is de ondergrond die de gebruikte typeannotaties aankunnen. Er zijn geen sleutels, accounts
of databases nodig: zowel de luchtfoto van PDOK als de registraties van Amsterdam zijn open
en worden tijdens de run opgehaald.

```bash
git clone https://github.com/ResultManagersWouter/aerial_classification_assets.git
cd aerial_classification_assets
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-ml.txt      # alleen voor --modellen
```

Werk je in PyCharm, wijs dan `.venv` aan als interpreter (Settings, Project, Python
Interpreter, Add Local Interpreter, Existing). Daarna draait `main.py` met de groene
knop.

Controleer of de verbindingen werken, dit haalt niets zwaars op:

```bash
.venv/bin/python -m luchtfoto_objecten gebieden   # de gebieden uit config/gebieden.yaml
.venv/bin/python -m luchtfoto_objecten lagen      # de luchtfotolagen en registraties
```

Doe dan een eerste analyse. Begin klein, en houd er rekening mee dat de eerste run van een
gebied alle tegels moet downloaden; een tweede run over hetzelfde gebied komt uit de cache
in `data/cache/` en is veel sneller.

```bash
.venv/bin/python main.py
```

Dat draait het voorbeeldgebied in Centrum, ongeveer 37 hectare, en dat kost een paar
minuten. Kijk daarna in `output/centrum_vergelijking/`: de GeoPackage open je in QGIS, en
`afwijkingen.csv` is de lijst waar het om gaat. Wil je begrijpen wat je ziet, lees dan
hieronder "Wat eruit komt", en vooral "Beperkingen", want een deel van de signalen is
ruis met een bekende oorzaak.

## Gebruiken

`main.py` is de ingang. Je geeft een gebied mee in Rijksdriehoek (EPSG:28992), en de
uitvoer komt in `output/<naam>/`.

```bash
python main.py                                    # het voorbeeldgebied in Centrum
python main.py --extent 121641.1653 122215.9413 486408.2909 487051.9774 --naam centrum
python main.py --bbox 121641 486408 122215 487051 --naam centrum
python main.py --grens data/aoi/buurt.geojson --naam buurt
python main.py --gebied noord_vliegenbos          # een gebied uit config/gebieden.yaml
python main.py --gebied noord_vliegenbos --modellen   # elk model als eigen laag
python main.py --gebied noord_noorderpark --bomen     # boomkroon of groenvlak
```

Let op de volgorde van de coördinaten. QGIS toont een extent als xmin, xmax, ymin, ymax en
die neem je over achter `--extent`. De rest van dit project gebruikt xmin, ymin, xmax, ymax,
en dat is wat `--bbox` verwacht. Met `--grens` geef je een bestand mee met de echte
gebiedsgrens (GeoJSON, Shapefile, GeoPackage); dat mag elke polygon zijn, niet alleen een
rechthoek, en de uitvoer wordt dan op die grens geknipt in plaats van op de rechthoek
eromheen.

Daarnaast is er de CLI van het pakket zelf, handig om alleen een deel te draaien:

```bash
python -m luchtfoto_objecten gebieden
python -m luchtfoto_objecten analyse --gebied centrum_plantage
python -m luchtfoto_objecten analyse --bbox 122100 486450 122500 486800 --zoom 16
python -m luchtfoto_objecten registratie --gebied noord_ndsm
```

De gebieden staan in `config/gebieden.yaml`, in RD-coördinaten. Begin klein. Een vlak van
400 bij 350 meter is op zoom 15 ruim tweehonderd tegels en draait in een halve minuut. Heel
stadsdeel Centrum staat er ook in, maar dat zijn ruim achtduizend tegels.

De optionele SAM-module staat in `requirements-sam.txt` en is alleen nodig als je contouren
wilt laten aanscherpen door Segment Anything.

## Wat eruit komt

`main.py` schrijft naar `output/<naam>/`: een GeoPackage met alle lagen, `afwijkingen.csv`
met de vlakken die niet kloppen, en `samenvatting_per_thema.csv`. Met `--modellen` komen
`modelvergelijking.csv` en `kenmerkbelang.csv` erbij. De CLI van het pakket schrijft naar
`data/uitvoer/<gebied>/`, en levert daar ook de losse GeoJSON-lagen, de luchtfoto als
GeoTIFF in RD en een samenvatting in JSON. Beide mappen staan in `.gitignore`.

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

## Modellen vergelijken

`python main.py --modellen` zet de huidige drempels naast een aantal lerende modellen.
Twee losse binaire vragen, elk op de opname die erbij hoort, met de BGT als referentie en
het westelijke deel van het gebied als trainingsgebied en het oostelijke deel als toets.
De code staat in `luchtfoto_objecten/modelvergelijking.py`.

Bedenk goed wat er gemeten wordt. De registratie is juist het bestand dat we willen
controleren, dus dit meet overeenstemming, geen waarheid. Een model dat hoger scoort leest
het beeld beter op de plekken waar de registratie klopt, en dan is de rest van het verschil
een echt signaal in plaats van modelruis.

Kijk niet naar IoU. Binnen de beheerkaart is het maaiveld in de binnenstad voor ruim
negentig procent verharding, en een model dat overal "verharding" roept haalt daar een IoU
van 0,93 zonder iets te doen. Daarom staat `altijd_positief` als bodem in de tabel, wordt
er gebalanceerd getraind en sorteren we op MCC.

Op het gebied `--extent 121641.1653 122215.9413 486408.2909 487051.9774` kwam dit eruit:

| klasse | model | MCC | gebalanceerd | recall | precisie |
|---|---|---|---|---|---|
| verharding | gradient_boost | 0,105 | 0,598 | 0,603 | 0,948 |
| verharding | randomforest | 0,102 | 0,593 | 0,641 | 0,946 |
| verharding | logistisch | 0,090 | 0,585 | 0,528 | 0,948 |
| verharding | altijd_positief | 0,000 | 0,500 | 1,000 | 0,925 |
| verharding | regels (huidig) | -0,032 | 0,473 | 0,726 | 0,920 |
| groen | regels (huidig) | 0,118 | 0,736 | 0,805 | 0,034 |
| groen | gradient_boost | 0,016 | 0,506 | 0,018 | 0,037 |
| groen | randomforest | 0,013 | 0,504 | 0,012 | 0,035 |

Twee dingen om te weten voor je hier verder aan werkt. De verhardingsdrempels scoren op
dit gebied slechter dan kansniveau: ze noemen 73 procent van de pixels verharding en
verwerpen maar 22 procent van wat het niet is. En het wisselen van model levert weinig op,
alles blijft rond een MCC van 0,1 hangen. De bottleneck zit niet in het model maar in de
referentie: de BGT registreert hier 1,4 procent van het maaiveld als begroeid terwijl de
zomerfoto ruim dertig procent groene pixels laat zien, want boomkronen hangen over
geregistreerde verharding. Wat een kroon boven een rijbaan hoort te zijn is dus eerst een
definitiekwestie en pas daarna een modelkeuze.

Het kenmerkbelang wijst dezelfde kant op: textuur op 1 en 3 meter zijn voor beide klassen
de zwaarste kenmerken, kleur weegt nauwelijks mee. De informatie zit in de omgeving van een
pixel, en dat is precies wat een model dat pixel voor pixel kijkt slecht ziet. Een
segmentatiemodel met ruimtelijke context, SAM of een kleine U-Net op tegels, is een
kansrijkere volgende stap dan nog een classificatie per pixel. Draai de vergelijking ook op
een gebied met meer variatie, zoals `noord_noorderpark` of `centrum_plantage`; in een blok
dat voor 93 procent verhard is valt er weinig te meten.

## De groendrempel, en waarom Otsu hier tegenwerkt

Met `--modellen` krijgt elk model een eigen laag in de GeoPackage, `detectie_groen_<model>`
en `detectie_verharding_<model>`, allemaal met dezelfde nabewerking en op hetzelfde
analysevlak. Zo kun je in QGIS over de foto kijken welk model het beeld volgt in plaats van
af te gaan op een getal. Op het Vliegenbos, een gebied dat vrijwel geheel bos is, kwam
hier dit uit, op een analysevlak van 145.727 m2:

| model | groen gevonden | aandeel analysevlak |
|---|---|---|
| kmeans | 133.823 m2 | 92% |
| exg_ondergrens | 126.932 m2 | 87% |
| hsv_groen | 63.043 m2 | 43% |
| regels (huidig) | 51.688 m2 | 36% |
| exg_maaiveld | 45.995 m2 | 32% |

De huidige regels vinden dus ongeveer een derde van wat er staat. De oorzaak is Otsu.
`_groendrempel` neemt `max(otsu, exg_ondergrens)`, en die ondergrens beschermt alleen
tegen een te lage drempel, nooit tegen een te hoge. Otsu zoekt de scheiding tussen de twee
grootste pieken in het histogram, en in een gebied dat vooral groen is liggen die twee
pieken allebei ín het groen: schaduwkroon tegen zonnige kroon. De drempel komt dan op
0,100 terwijl de ondergrens 0,035 is, en alles wat donkerder groen is valt af. Het helpt
niet om Otsu alleen op het maaiveld te doen, dat werd hier zelfs 0,109.

## Boomkroon of groenvlak

Boomkronen langs de weg komen als wolkjes groen uit de detectie terwijl de verharding
eronder gewoon klopt. Met `--bomen` wordt onderzocht of die twee te scheiden zijn, geijkt
op de stamlocaties uit het bomenregister van Amsterdam (`bomen/stamgegevens`).

De kenmerken wijzen de goede kant op. Op het Noorderpark is een kroon groter (28 tegen 15
m2), ronder (compactheid 0,44 tegen 0,35), breder (3,5 tegen 2,6 m) en ruwer (textuur
0,029 tegen 0,025) dan een groenvlak. Maar de verdelingen overlappen zo sterk dat een
model op die kenmerken blijft steken op 69 procent juist, terwijl je met altijd de
grootste klasse roepen al op 61 procent zit. De F1 op boomkronen is 0,59. Op beeldkenmerken
alleen is het dus niet betrouwbaar te doen, zeker niet op de 25cm-zomeropname waarop de
kroonstructuur nauwelijks te zien is.

Twee routes die wel werken. De goedkoopste is het bomenregister zelf: Amsterdam weet waar
zijn bomen staan, inclusief hoogteklasse, dus je kunt kronen maskeren met een buffer om de
stam in plaats van ze uit pixels af te leiden. De stevigste is hoogte. AHN staat open via
de WCS van PDOK met `dsm_05m` en `dtm_05m`, en het verschil daartussen is de objecthoogte:
een kroon zit op 6 tot 20 meter, een gazon op nul. Dat scheidt wat kleur en vorm niet
kunnen scheiden.

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
