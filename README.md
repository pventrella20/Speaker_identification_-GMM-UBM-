# Speaker_identification_-GMM-UBM- System
Il sistema ha diversi requisiti, librerie da installare per il corretto funzionamento:
  - sklearn
  - python_speech_features
  - pydub
  - joblib
  - scipy.io
  - matplotlib.pyplot
  - pandas
  - seaborn

Il sistema lavora con file .WAV, si consiglia di lanciare il file 'convert.sh' presente in alcune cartelle per la conversione nel suddetto formato 
di file .ogg o .mp3 e il ricampionamento a 16kHz.

La cartella ''data'' contiene diverse cartelle al suo interno, il cui utilizzo di ognuna è descritto di seguito:
  - ''gmm_dataset'': deve contenere i file utilizzati per l'addestramento del sistema (necessari per leggere i nomi degli speakers)
  - ''model'': in questa cartella sono presenti i modelli .pkl degli speaker memorizzati
  - ''temp'': inserire in questa cartella i file da splittare
  - ''splitted'': in questa cartella verranno creati gli split a partire dagli audio presenti nella cartella ''temp''
  - ''test'': inserire qui i file di testing
  - ''ubm_dataset'': vi si trovano i file necessari all'addestramento della UBM<br>
NOTA: i nomi dei file di training devono seguire la seguente sintassi -> #_nomespeaker_cognomespeaker_note<br>
      per # si intende un identificativo numerico o letterale,<br>
      per note si intende un qualsiasi testo (numeri o lettere) utile per l'utente (ad esempio il personaggio imitato)<br>
      es. AA_alberto_angela_0 oppure 01_maurizio_crozza_renzi
      
## **SETUP**
Il sistema si avvia lanciando ''main.py'', e chiede se deve essere lanciato in modalità ''train'' [scrivendo 'train'] o ''test'' [scrivendo 'test'], 
ricordando di inserire i file di training in ''data/gmm_ubm'' e quelli di testing in ''data/test''.

## **SPLITTING**
In ogni caso verrà chiesto se è necessario splittare i file [yes/no], in caso affermativo inserire i file nella cartella ''data/temp'' e proseguire
inserendo la lunghezza dei segmenti; si troveranno i file splittati nella cartella ''data/splitted''. Bisognerà spostare poi manulamente i file nella
cartella di destinazione (se di training o di testing).

## **TRAINING**
- verrà chiesto di inserire il numero di componenti Gaussiane da utilizzare (una potenza di 2)
- dopodichè inizierà automaticamente il training della UBM a partire dai file presenti in ''data/ubm_dataset'', e verranno adattati i singoli modelli
  a partire dalle features estratte dai file presenti in ''data/gmm_dataset''
- i modelli verranno memorizzati in ''data/model'', ed eventuali modelli già presenti con lo stesso nome verranno sovrascritti.
- dopo il training viene avviata automaticamente la fase di testing

## **TESTING**
- il testing inizierà automaticamente dopo la richiesta di split, a partire dai file presenti in ''data/test''
- durante il testing viene fornito un feedback sull'andamento del processo, ed alla fine elencati gli speaker predetti per ogni file di test
- alla fine del testing verrà visualizzata la matrice di confusione prodotta

## **ESEMPIO**
- si consiglia di iniziare a sperimentare sui dati già presenti nelle rispettive cartelle, lanciando ''main.py'', scrivendo 'test', digitando 'no' 
  lo splitting e facendo partire il sistema sui file di 1 minuto presenti nella cartella ''data/test'', poiché far partire un nuovo training sovrascrirebbe
  i modelli già presenti.
