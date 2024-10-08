# DN 11. Reševanje PDE z metodo Galerkina (Reševanje Poissonove enačbe)

Še zadnji okus metode za reševanja parcialnih diferencialnih enačb. Tokrat se bomo lotili reševanja Poissonove enačbe z metodo Galerkina. Pri metodi Galerkina se rešitve aproksimira z neko kombinacijo baznih funkcij, ki jih določimo z znanjem o fiziki problema in robnih pogojih.


## Navodila
Naloga želi, da rešimo Poissonovo enačbo v 2D za polkrožno cev. Cilj je izračunati koeficient pretoka $C$ skozi takšno cev. Poglejmo si tudi kako je odvisna rešitev od števila baznih funkcij, torej števila členov v indeksih $n$ in $m$.

## Napotki
Hah pri tej nalogi sem se veliko igral s kreativnim risanjem slik v malo drugačnem slogu. Profesorju se je dopadel old school stil, ampak je pripomnil, da mu je osebno malo preveč 
*Matrix-Like* (kot v filmu Matrix). Ker to objavljam dolgo po oddaji naloge nimam nekaj smiselnih napotkov. Če se kdo želi igrati, je s metodo, ki jo preučujemo možno rešiti Poissonovo enačbo še za kakšno drugo obliko cevi.

## Kar sem jaz naredil
**Tu je verjetno tisto kar te najbolj zanima**. 

<details>
  <summary>Standard Disclaimer</summary>
  Objavljam tudi kodo. Ta je bila tokrat v svojem repozitoriju od začetka, ker sem teh zadnjih nekaj nalog opravljal med poletjem. Koda bi morala biti razmeroma pokomentirana, sploh v kasnejših nalogah. 
  
</details>

Vseeno pa priporočam, da si najprej sam poskusiš rešiti nalogo. As always za vprašanja sem na voljo.


* [**Poročilo DN11**](https://pengu5055.github.io/fmf-pdf/year3/mfp/Marko_Urban%C4%8D_11.pdf)
* [**Source repozitorij DN11**](https://github.com/pengu5055/mfp11)

Priznam, da zna biti source repozitorij nekoliko kaotičen. Over time sem se naučil boljše prakse. Zdi se mi, da je tole glavni `.py` file.

* [**src.py**](https://github.com/pengu5055/mfp11/blob/main/src.py)

## Citiranje
*Malo za šalo, malo za res*.. če želiš izpostaviti/omeniti/se sklicati ali pa karkoli že, na moje delo, potem ga lahko preprosto citiraš kot:

```bib
@misc{Urbanč_mfpDN11, 
  title={Reševanje PDE z metodo Galerkina}, 
  url={https://pengu5055.github.io/fmf-pages/year3/mfp/dn11.html}, 
  journal={Marko’s Chest}, 
  author={Urbanč, Marko}, 
  year={2023}, 
  month={Oct}
} 
```
To je veliko boljše kot prepisovanje.
