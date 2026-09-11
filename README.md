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
```

Daarmee draait `python main.py` volledig; er is verder niets nodig. Twee losse bestanden
zijn optioneel en hoef je alleen te installeren als je die stap gebruikt:

| bestand | waarvoor | wanneer |
|---|---|---|
| `requirements.txt` | de hele pijplijn | altijd |
| `requirements-ml.txt` | scikit-learn | voor `--modellen` en `--bomen` |
| `requirements-sam.txt` | torch en Segment Anything, ruim 2 GB | alleen voor contourverfijning met SAM |

Twee dingen die verwarren: de module `yaml` komt uit het pakket `pyyaml`, dat staat er dus
gewoon in, en `pyogrio`, dat de GeoPackages wegschrijft, komt automatisch mee met
`geopandas`. Je hoeft die twee niet apart te installeren.

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

## Alle commando's op een rij

Eén ingang, `main.py`, en een vlag bepaalt wat je krijgt. Zonder vlag draait de gewone
analyse. De vlaggen zijn te combineren, behalve `--sweep` en `--jaren`, die alleen hun
eigen werk doen en daarna stoppen.

| commando | wat het doet | wat je krijgt in `output/<naam>/` |
|---|---|---|
| `python main.py --gebied X` | detectie plus vergelijking met de registratie | `X.gpkg` met de signalering, `afwijkingen.csv`, `samenvatting_per_thema.csv` |
| `... --modellen` | elk detectiemodel als eigen laag, en een score tegen de BGT | extra lagen `detectie_groen_<model>` en `detectie_verharding_<model>`, plus `modeloverzicht.csv`, `modelvergelijking.csv`, `kenmerkbelang.csv` |
| `... --bomen` | is het gevonden groen een kroon of maaiveld, geijkt op het bomenregister | laag `groen_boom_of_vlak`, plus `boom_of_vlak.csv` en `boomherkenning_score.csv` |
| `... --sweep` | tien parameterinstellingen per klasse, om zelf te beoordelen | vier losse bestanden: `groen.gpkg`, `verharding.gpkg`, `bomen.gpkg`, `infrarood.gpkg`, elk met `sweep_*.csv` |
| `... --jaren N` | groen en verharding over de laatste N jaargangen | `meerjaren.gpkg` met een laag per klasse per jaar, `meerjaren_overzicht.csv`, `meerjaren_verschillen.csv`, `meerjaren_per_asset.csv` |

Het gebied geef je op één van vier manieren op, en de rest van de vlaggen werkt er
hetzelfde mee:

| vlag | volgorde | voorbeeld |
|---|---|---|
| `--gebied` | naam uit `config/gebieden.yaml` | `--gebied noord_vliegenbos` |
| `--extent` | xmin xmax ymin ymax, zoals QGIS toont | `--extent 121641 122215 486408 487051` |
| `--bbox` | xmin ymin xmax ymax, zoals dit project | `--bbox 121641 486408 122215 487051` |
| `--grens` | bestand met een polygon, ook niet-rechthoekig | `--grens data/aoi/buurt.geojson` |

Verder zijn er `--naam` voor de uitvoermap, en `--zoom` voor de resolutie (15 is 10,5 cm
per pixel, 16 is 5,25 cm en viermaal zoveel tegels).

In elke GeoPackage zit ook de luchtfoto waarop de classificatie rust, als rasterlaag
`luchtfoto_<laagnaam>`. Je opent dus één bestand in QGIS en hebt beeld en vlakken bij
elkaar, inclusief het infrarood.

Naast `main.py` bestaat de CLI van het pakket zelf, handig om alleen een deel te draaien:

```bash
python -m luchtfoto_objecten gebieden      # de 49 gebieden
python -m luchtfoto_objecten lagen         # luchtfotolagen en registraties
python -m luchtfoto_objecten registratie --gebied noord_ndsm
```

## Over de jaren heen vergelijken

`--jaren 4` legt de laatste vier jaargangen naast elkaar. Drie dingen maken dat lastig, en
die worden alle drie aangepakt.

Elke jaargang heeft zijn eigen kleurzweem, dus een vaste drempel die op 2026 klopt mist in
2023 de helft. De drempel wordt daarom per jaargang opnieuw op dezelfde registratie geijkt.
Op het Noorderpark levert dat vier heel verschillende drempels op, van +0,004 in 2026 tot
-0,037 in 2023, terwijl de overeenkomst met de registratie steeds rond de 0,75 tot 0,87 IoU
blijft. Wat je vergelijkt zijn dus detecties die elk even goed zijn afgeregeld.

Niet elk jaar heeft beide opnamen. Daarom wordt gekeken welke modaliteit álle gekozen jaren
hebben, en daarop wordt vergeleken; infrarood als dat kan, anders kleur. De gebruikte
opname staat per jaar in `meerjaren_overzicht.csv`, zodat je kunt zien waar een sprong
vandaan komt.

En bomen wisselen met het seizoen: in volle bloei is de kroon veel groter dan in het
voorjaar. Kroonoppervlak van jaar tot jaar naast elkaar leggen meet dan vooral de
bladstand. Daarom staat het bladaandeel per jaargang in het overzicht en ook naast elk
verschil, zodat een kleinere kroon in een kaal jaar niet als gekapte boom wordt gelezen.

In `meerjaren_per_asset.csv` staat per geregistreerd object welk deel ervan elk jaar op de
foto terug te zien is, met de verandering tussen het eerste en het laatste jaar. Dat is de
lijst om op te sorteren als je wilt weten welk object echt veranderd is.

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
python main.py --gebied noord_vliegenbos --sweep      # tien instellingen per klasse
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

In `config/gebieden.yaml` staan 49 gebieden klaar, verspreid over alle stadsdelen: pleinen
in het centrum, stadsparken, sportvelden, een begraafplaats, nieuwbouweilanden, bedrijven-
terrein en landelijk gebied. Dat is met opzet gevarieerd, want een detectiedrempel die op
de Dam werkt hoeft in het Vliegenbos nog niets te doen. `python -m luchtfoto_objecten
gebieden` toont de lijst, en dezelfde vlakken staan als polygon in `config/gebieden.geojson`
om in QGIS te bekijken of mee te geven met `--grens`.

De middelpunten komen uit de locatieserver van PDOK, dus ze liggen waar ze horen. Begin
klein: de meeste gebieden zijn 300 tot 500 meter en draaien in een halve minuut. Heel
stadsdeel Centrum staat er ook in als `centrum_volledig`, maar dat zijn ruim achtduizend
tegels.

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

## Instellingen uitproberen

`--sweep` draait tien instellingen per klasse en zet ze in drie aparte GeoPackages, zodat
je er in QGIS met de luchtfoto ernaast doorheen kunt lopen:

    output/<gebied>/groen.gpkg        tien manieren om vegetatie te vinden
    output/<gebied>/verharding.gpkg   tien manieren om verharding te vinden
    output/<gebied>/bomen.gpkg        tien manieren om kroon van maaiveld te scheiden

De lagen zijn genummerd zodat ze op volgorde staan, de naam zegt wat er anders is, en
naast elk bestand staat een `sweep_*.csv` met de instelling en de opbrengst. Alle
varianten krijgen dezelfde nabewerking en hetzelfde analysevlak, dus wat je ziet is het
verschil van de parameter.

Op het Vliegenbos loopt groen van 45.995 m2 (Otsu op het maaiveld) tot 138.847 m2 (vaste
drempel 0.020), op een analysevlak van 145.727 m2. Verharding loopt van 80.336 tot 144.722
m2, en de kroon van 3.404 tot 46.313 m2. Er valt dus genoeg te kiezen, en juist daarom is
het beeld de scheidsrechter en niet een getal.

## Groen op het maaiveld tegenover de kroon

De groendetectie is vooral goed in bomen, want een kroon is het felste groen op de foto.
Een grasveld met bomen erop komt er daardoor uit als een verzameling kronen in plaats van
als het grasveld dat het is.

De twee opnamen lossen dat samen op. De zomeropname laat zien waar blad hangt. De
8cm-ortho is in het vroege voorjaar gevlogen met kale bomen en laat dus de grond ónder die
kroon zien, dus de verhardingsdetectie daarop vertelt of die ondergrond hard of zacht is.
Gras onder een bomenrij is in de zomer onzichtbaar maar in het voorjaar gewoon te zien.

Dat levert twee lagen op die elkaar mogen overlappen, want zo is het ook: de kroon hangt
boven de grond.

- `detectie_boomkroon`, groen op de zomerfoto met de ruwe textuur van blad en takken
- `detectie_groenstrook`, begroeid maaiveld, ook waar een kroon eroverheen hangt

Het groensignaal is leidend, de onverharde ondergrond telt mee als steun en ruwe textuur
pleit ertegen. De gewichten staan onder `groenstructuur` in `config/parameters.yaml`.

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
