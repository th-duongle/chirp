# Bird Species Classification Using Convolutional Neural Networks

## Overview

This project investigates the use of Convolutional Neural Networks (CNNs) for automated bird species classification using spectrogram representations of bird vocalizations. The project explores both binary and multiclass classification problems using deep learning techniques applied to audio spectrograms treated as image-like inputs.

The primary objectives of this project are to:

* Classify bird species from audio recordings using CNN architectures
* Compare binary and multiclass CNN performance
* Investigate the effects of class imbalance and spectrogram augmentation
* Explore overfitting, regularization, and architecture improvements
* Evaluate CNN performance on previously unseen real-world audio clips

The project was implemented in Python using TensorFlow/Keras and standard audio processing libraries.

---

## Bird Species Included

* American Crow (`amecro`)
* American Robin (`amerob`)
* Bewick’s Wren (`bewwre`)
* Black-capped Chickadee (`bkcchi`)
* Dark-eyed Junco (`daejun`)
* House Finch (`houfin`)
* House Sparrow (`houspa`)
* Northern Flicker (`norfli`)
* Red-winged Blackbird (`rewbla`)
* Song Sparrow (`sonspa`)
* Spotted Towhee (`spotow`)
* White-crowned Sparrow (`whcspa`)

# Libraries Used

## Core Libraries

```python
numpy
pandas
matplotlib
seaborn
h5py
librosa
tensorflow / keras
scikit-learn
```

---

# Key Takeaways

* CNNs are highly effective for spectrogram-based bird classification.
* Class imbalance significantly impacts multiclass performance.
* Spectrogram augmentation substantially improves generalization.
* More sophisticated CNN architectures improve performance but greatly increase training time.
* Real-world environmental audio presents additional challenges due to overlapping bird vocalizations.

---

# Future Improvements

Potential future work includes:

* Transfer learning using BirdNET or EfficientNet
* Multi-label bird classification
* CNN-RNN hybrid architectures
* Larger and more diverse datasets
* Advanced audio augmentation techniques
* Real-time bird detection systems

---

# References

* TensorFlow / Keras Documentation
* Librosa Audio Processing Library
* SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
* BirdNET: A deep learning solution for avian diversity monitoring
