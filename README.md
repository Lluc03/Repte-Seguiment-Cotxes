# Repte Cotxes

**Autor:** Lluc Verdaguer Macias  
---

## Descripció

Aquest projecte implementa un **pipeline complet de detecció, seguiment i recompte de vehicles** a partir de vídeo.  
L’objectiu principal és aconseguir un **sistema eficient i en temps real**, capaç de detectar, seguir i comptar cotxes segons la seva direcció de moviment (nord-sud, est-oest, etc.).

El pipeline està format per tres mòduls principals:

1. **Detector (YOLOv8n)**
2. **Tracker (Centroid / Direcció / Predicció / HOG)**
3. **Comptador de vehicles**


El temps total del pipeline per *frame* és d’uns **32.37 ms**, el que permet **processar vídeo en temps real (30 fps)**.

---

## 1. Detector — YOLOv8n

- **Model:** `YOLOv8n.pt`  
- **Classe utilitzada:** `"car"`  
- **Temps mitjà per frame:** `31.14 ms`

Permet detectar cotxes de manera ràpida i precisa, aprofitant que YOLOv8 ja inclou aquesta classe entrenada.

---

## 2. Comptador de Vehicles

El comptador utilitza:

- Una línia horitzontal de referència  
- L’historial de **centroids** dels vehicles detectats  

### Funcionament:
1. Es registra el **centroid** de cada vehicle.  
2. Es calcula la direcció del moviment.  
3. Quan un vehicle travessa la línia:
   - **De dalt a baix → +1 Sud**  
   - **De baix a dalt → +1 Nord**

---

## 3. Tracker (Seguiment)

Diverses versions del *tracker* han estat desenvolupades per millorar la precisió i robustesa:

| Versió | Mètode | Temps mitjà (ms) | Resultats destacats |
|--------|--------|------------------|---------------------|
| v1 | Centroid Tracker | 0.0 | Ràpid però confon identitats en creuaments |
| v2 | Centroid + Direcció | 0.0 | Millora el sentit de circulació |
| v3 | Centroid + Direcció + Predicció | 0.0 | Manté identitat si el vehicle desapareix temporalment |
| v4 | Centroid + Direcció + Predicció + HOG | **3.35** | Reconeixement més robust basat en gradients locals |

---

### Què fa el HOG (Histogram of Oriented Gradients)?

Afegeix una capa de robustesa al tracker.

- Analitza la **forma i textura** en lloc del color  
- Calcula la **direcció i magnitud** del canvi d’intensitat lumínica entre píxels  
- Permet distingir vehicles similars o parcialment ocults  

---

## Rendiment Global

| Mòdul | Temps mitjà per frame |
|--------|------------------------|
| Detector (YOLOv8n) | 31.14 ms |
| Tracker (HOG) | 3.35 ms |
| Comptador | 0.0 ms |
| **Total** | **32.37 ms** |

**Apta per temps real (30 fps)**

---

## Resultats

| Escenari | Ground Truth | Predicció Tracker v4 |
|-----------|---------------|---------------------|
| Short | ↑6 / ↓2 | ↑6 / ↓2 |
| Middle | ↑5 / ↓7 | ↑5 / ↓7 |
| Shadow | ↑3 / ↓10 | ↑3 / ↓10 |
| Long 1 | ↑8 / ↓24 | ↑10 / ↓26 |

**Total vehicles (Short):**  
- **Real:** 74  
- **Detectats:** 87

---

## Conclusions

- Detecció i seguiment en temps real  
- Implementació correcta del recompte nord-sud  
- Escalable a est-oest  
- Alguns *frames* no detecten tots els vehicles  

### Possibles millores:
- Fer servir només **HOG Tracker**, descartant centroid/direcció/predicció per augmentar precisió  
- Millorar la configuració del HOG per més *accuracy*  
- Augmentar el *confidence level* del YOLO per reduir falsos positius  

---

## Referències

- **Centroid Tracker:**  
  [Simple Object Tracking with OpenCV – PyImageSearch](https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)

- **People Tracker amb YOLOv12:**  
  [People Tracker – PyImageSearch](https://pyimagesearch.com/2025/07/14/people-tracker-with-yolov12-and-centroid-tracker/)

- **HOG Tracker:**  
  [Object tracking using HOG – GitHub](https://github.com/imenebak/Object-tracking-using-Hog)

---


