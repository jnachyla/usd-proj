# Project USD

Implementation of TD3 algorithm based on DDPG implementation in keras:

https://keras.io/examples/rl/ddpg_pendulum/

Tested on Pendulum(https://gymnasium.farama.org/environments/classic_control/pendulum/) environment.
Evaluation over 100 episodes: Mean -156.063194 with standard deviation: 2.323086.

![alt text](Isolated.png "Title")


## Tworzenie wirtualnego środowiska
W celu utworzenia środowiska wirtualnego należy wpisać w terminalu komendę:
```bash
$ python3 -m venv nazwa_srodowiska
```
W kolejnym kroku należy aktywować środowisko wirtualne. 
Na Linuxie służy do tego komenda:
```bash
$ source nazwa_srodowiska/bin/activate
```
Na systemach Windows:
```bash
> .\nazwa_srodowiska\Scripts\activate.bat
```
Nazwa środowiska pojawi się w nawiasach na początku linii. Środowisko jest gotowe. 

Źródło: https://kamil.kwapisz.pl/srodowiska-wirtualne/#Pipenv
## Instalacja wymaganych bibliotek
Wymagane biblioteki i ich wersje znajdują się w pliku requirements.txt.

Po utworzeniu wirtualnego środowiska można je zainstalować poleceniem:
```bash
pip3 install -r requirements.txt
```
