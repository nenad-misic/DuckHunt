# DuckHunt
Projektni zadatak iz predmeta Soft computing, 2019/2020


## Članovi tima:
   Nenad Mišić, SW-31/2016, Grupa 2


## Asistent:
   Dragan Vidaković

## Definicija problema:
Kreiranje kopije popularne Nintendo igre "Duck Hunt" tako da bude kontrolisana preko mobilnog uređaja povezanog na lokalnu mrežu sa računarom. Kupovinom tradicionalne verzije igre dobijao se i kontroler kojim se ciljalo prema ekranu televizora i tako ubijale ptice. Više o igri se može naći na sledećem [linku](https://en.wikipedia.org/wiki/Duck_Hunt).

Ideja je napraviti program koji procesira sliku koju dobija od mobilnog telefona, zaključuje da li je telefon uperen prema nekoj od ptica i, ukoliko jeste, ubija pticu. 

## Korišćene tehnologije: 

* Python 2.7.1 i PyGame 1.9.1 za implementaciju igre
  * Koristi se već urađena open-source verzija igre: [link](https://www.pygame.org/project-Duck+Hunt+Remake-2045-.html)

* IP Webcam aplikacija za mobilni telefon
  * Koristi se android aplikacija namenjena za "home surveillance", koja podiže server na mobilnom telefonu i preko "endpointa" pruža pristup video streamu kao i trenutnoj fotografiji: [link](https://play.google.com/store/apps/details?id=com.pas.webcam)
 
* Python 3.6.1, Flask, PyAutoGUI, Keras i OpenCV za implementaciju glavnog programa


## Algoritam/algoritmi:
* Canny Edge Detector 
  * za pronalaženje konture ekrana
* HOG Feature Exctractor 
  * za ekstrakciju osobina slike (korišćen u kombinaciji sa KNN i SVM)
* KNN 
  * za klasifikaciju slika uz HOG
* SVM 
  * za klasifikaciju slika uz HOG
* CNN 
  * za klasifikaciju slika

## Skup podataka
S obzirom da je problem koji se rešava vrlo usko specifičan, skup podataka se mora ručno prikupiti i anotirati.
Slike će biti telefonom uslikani trenuci u igri, sa kojih prvo treba izdvojiti ekran. Zatim se na slikama ručno označe ptice kao regije od interesa. Nakon toga sledi podela slike na 15 kolona i 10 redova i čuvanje 2x2 blokova. Svaki 2x2 blok koji preseca regiju od interesa označavamo kao potencijalnu pticu, dok sve ostale označavamo kao sigurnu pozadinu. Poslednji korak je ručno pregledanje potencijalno pozitivnih slika i izbacivanje onih koje nisu odgovarajuće (recimo slike na kojima se vidi samo par piksela ptice). Koristi se 80 početnih slika iz kojih se dobija oko 2000 pozitivnih i 8000 negativnih slika. Nad njima se dalje vrši augmentacija i izdvaja "sample" negativnih slika kako bi obe klase imale isti broj članova.

Skup će biti podeljen na trening i test skup u razmeri 75:25.

## Metrika za merenje performansi:
Poređenje predložena tri algoritma međusobno.
Poredi se accuracy nad istim skupom podataka.

## Validacija rešenja:
Demonstriranje rešenja na odbrani projekta uživo.
