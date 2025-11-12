# 🛰️ Feature Engineering — Space-Track Satellite Data

## 1. 🧩 Features Statiques (catalogue général `active_satellites.csv`)

Ces variables décrivent la **configuration orbitale instantanée** d’un satellite.  
Elles sont principalement utilisées pour le **profiling global** et la **détection d’anomalies statiques**.

| Catégorie | Feature | Description | Intérêt pour le ML |
|------------|----------|-------------|--------------------|
| **Géométrie orbitale** | `INCLINATION` | Inclinaison orbitale (°) | Indique le type d’orbite (polaire, équatoriale, etc.) |
| | `ECCENTRICITY` | Excentricité de l’orbite | Mesure la circularité orbitale |
| | `ARG_OF_PERICENTER` | Argument du péricentre (°) | Position du point le plus proche de la Terre |
| | `RA_OF_ASC_NODE` | Ascension droite du nœud ascendant | Orientation du plan orbital |
| | `MEAN_ANOMALY` | Anomalie moyenne (°) | Permet d’estimer la position sur l’orbite |
| **Dynamique orbitale** | `MEAN_MOTION` | Révolutions par jour | Indice du rayon orbital |
| | `MEAN_MOTION_DOT`, `MEAN_MOTION_DDOT` | Dérivées du mouvement moyen | Indiquent une dérive orbitale potentielle |
| | `REV_AT_EPOCH` | Révolution totale à l’époque | Sert à construire la chronologie orbitale |
| **Perturbations atmosphériques** | `BSTAR` | Coefficient de freinage atmosphérique | Reflète l’influence de la densité atmosphérique |
| **Typologie** | `CLASSIFICATION_TYPE`, `EPHEMERIS_TYPE` | Type d’objet (payload, debris, etc.) | Permet d’exclure les objets non pertinents |
| **Variables dérivées (à calculer)** | `altitude_estimated` | Calculée via la loi de Kepler (voir formules) | Approximation physique de l’altitude orbitale |
| | `orbit_type` | Catégorie (LEO / MEO / GEO) | Regroupement utile pour la segmentation |
| | `orbital_period_min` | 1440 / `MEAN_MOTION` | Temps d’une orbite complète en minutes |
| | `drag_index` | `BSTAR × ECCENTRICITY` | Indicateur de frottement atmosphérique |
| | `energy_index` | 1 / (2 × `MEAN_MOTION`²) | Approximation d’énergie orbitale spécifique |

> 🎯 **Objectif :** identifier les satellites dont les caractéristiques sont physiquement incohérentes avec leur classe orbitale.

---

## 2. 🔁 Features Dynamiques (série temporelle `STARLINK-3874_52365_data.csv`)

Ces variables exploitent les **variations temporelles** des paramètres orbitaux.  
Elles sont utilisées pour la **détection d’anomalies évolutives** (par exemple : perte d’altitude, manœuvre, dérive, etc.).

| Catégorie | Feature | Description | Interprétation |
|------------|----------|-------------|----------------|
| **Variation de position orbitale** | `delta_semimajor_axis` | Différence de l’axe semi-majeur entre deux observations | Gain ou perte d’altitude |
| | `delta_apogee`, `delta_perigee` | Variation des altitudes extrêmes | Décroissance ou montée non prévue |
| | `delta_inclination` | Variation de l’inclinaison | Manœuvre ou dérive orbitale |
| | `delta_mean_motion` | Variation du mouvement moyen | Perturbation physique |
| **Indices de stabilité (rolling features)** | `std_inclination_7d` | Écart-type glissant sur 7 jours | Indice de stabilité orbitale |
| | `std_eccentricity_7d` | Écart-type de l’excentricité sur 7 jours | Détecte une orbite instable |
| **Relations physiques** | `orbit_energy = -GM/(2a)` | Énergie orbitale théorique | Doit rester stable pour un satellite sain |
| | `delta_energy` | Variation d’énergie orbitale | Baisse soudaine = anomalie |
| **Événements potentiels** | `is_decay_event` | `DECAYED == 1` | Indique une désintégration orbitale |
| | `high_drag_event` | `BSTAR` > seuil | Frottement atmosphérique élevé |
| **Agrégats temporels** | `mean_altitude_weekly` | Moyenne glissante de l’altitude | Indique la tendance générale |
| | `trend_altitude` | Pente linéaire sur n observations | Chute lente → alerte précoce |

> 🎯 **Objectif :** alimenter un modèle d’apprentissage pour la détection d’anomalies temporelles (IsolationForest, Autoencoder, LSTM...).

---

## 3. 🧮 Formules Utiles

### 📘 Altitude estimée (en km)
\[
Altitude = \left( \frac{398600.4418}{(2\pi \cdot MEAN\_MOTION / 86400)^2} \right)^{1/3} - 6371
\]

> Constante gravitationnelle terrestre : **398600.4418 km³/s²**

---

### 🛰️ Classification orbitale (selon altitude moyenne)
| Type d’orbite | Altitude moyenne | Exemples |
|----------------|------------------|-----------|
| **LEO** (Low Earth Orbit) | < 2 000 km | Starlink, ISS |
| **MEO** (Medium Earth Orbit) | 2 000 – 35 000 km | GPS |
| **GEO** (Geostationary Orbit) | ≈ 35 786 km | Satellites TV |

---

### ⚙️ Indicateurs dérivés
- **Drag Index** = `BSTAR × ECCENTRICITY`  
  → Mesure l’influence combinée du frottement atmosphérique et de l’excentricité.  
- **Orbital Energy Index** = `1 / (2 × MEAN_MOTION²)`  
  → Approximation simplifiée de l’énergie orbitale.  
- **Variation Rate** = `ΔVariable / ΔTemps`  
  → Taux de changement entre deux observations successives.

---

## 4. 🎯 Sélection Finale des Features Candidates

| Type | Variables clés | Utilisation |
|-------|----------------|-------------|
| **Statique (catalogue)** | `INCLINATION`, `ECCENTRICITY`, `MEAN_MOTION`, `BSTAR`, `altitude_estimated`, `orbit_type`, `drag_index`, `orbital_period_min` | Pour le profiling global et la détection d’outliers |
| **Dynamique (Starlink)** | `delta_apogee`, `delta_perigee`, `delta_inclination`, `delta_mean_motion`, `std_inclination_7d`, `std_eccentricity_7d`, `delta_energy`, `trend_altitude`, `is_decay_event` | Pour les modèles de détection d’anomalies temporelles |
| **Meta / Classification** | `OBJECT_TYPE`, `orbit_class`, `DECAYED`, `CLASSIFICATION_TYPE` | Pour filtrer, labelliser ou segmenter les objets |

---

📘 **Résumé :**
Ce jeu de *features* fournit une représentation complète de l’état orbital et de son évolution dans le temps.  
Les variables statiques assurent la cohérence physique, tandis que les variables dynamiques mesurent la stabilité et les dérives orbitale — éléments essentiels pour construire un modèle d’anomaly detection fiable.
