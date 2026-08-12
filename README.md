# Spectra-FakeNet

**A dual-stream spatial and frequency network for deepfake detection. This is the research behind [DeepDetect V2](https://github.com/vendotha/DeepDetect-V2).**

Most deepfake detectors are trained purely on RGB pixels, and while they score well on the dataset they were trained on, that accuracy tends to fall away once they meet a manipulation method or a compression level they have not seen before. This paper works from a 2025 survey of the field to name the reason: spatial only detectors overfit to the pixel artifacts of one generator and have no way of picking up the signal that generative adversarial networks leave behind in the frequency domain. Spectra-FakeNet adds that signal back in, as a second learned stream running in parallel with the spatial branch.

[Paper PDF](./Spectra-FakeNet_Research_Paper.pdf) · [DeepDetect V2 (implementation)](https://github.com/vendotha/DeepDetect-V2)

## The problem

A model trained on one face swap tool loses accuracy the moment it is shown output from a different one, and it loses more once ordinary social media compression is applied, because the fine spatial artifacts the model learned to key on are exactly what compression removes first. This is a generalisation problem rather than an accuracy problem, and it is why a detector can look excellent on a benchmark leaderboard and still be unreliable once deployed. The survey this paper builds on also points out that very few detectors touch the frequency domain at all, even though the up sampling layers inside GAN generators leave a periodic, grid like signature in an image's spectrum that is largely invisible to the eye.

## The fix: a dual-stream network

Spectra-FakeNet processes a face crop as two parallel signals instead of one.

- **Spatial branch**: a compact CNN built from Xception style blocks, working directly on the RGB crop, looking for blending seams, warping, and unnatural texture
- **Frequency branch**: the same crop converted to grayscale, transformed with a 2D Discrete Cosine Transform, and compressed with a log magnitude step, so a second CNN can look for the grid pattern GAN up sampling leaves in the spectrum

The two branches never share weights. Combining them before feature extraction was tried and found to wash out the frequency signal rather than reinforce it, so fusion happens later: both branches are flattened, concatenated, passed through a dense layer with dropout, and classified with a single sigmoid unit.

## Results

Reported on the paper's evaluation set (80 held out crops, 44 real and 36 fake):

| Test | Result |
|---|---|
| Held out confusion matrix | Perfect separation, 0 false positives, 0 false negatives |
| Accuracy under Gaussian blur | 92%, vs 78% for a spatial only baseline |
| Visually convincing fakes caught by the frequency branch | 85% |

The confusion matrix result is reported honestly as a small sample sanity check rather than a generalisation claim. The paper is explicit that the evaluation set is small and single source, and lays out cross dataset validation against FaceForensics++, DFDC, and Celeb-DF as the next step before any broader claim can be made with confidence.

## Why the frequency branch matters

The blur test is the clearest evidence for the design. Blur suppresses the fine spatial texture a spatial only model depends on, but it leaves the coarse structure of the DCT grid artifact mostly intact, which is why the fused model's accuracy holds up noticeably better under degradation than a spatial only baseline's does. That gap is the whole argument for building the frequency signal into the network rather than treating it as a separate forensic check run on the side.

## Implementation

[DeepDetect V2](https://github.com/vendotha/DeepDetect-V2) is the working system built on this research: a Flask app with the same dual-stream spatial plus frequency pipeline, MTCNN face extraction, and a browser based interface for running detection on images and video.

## Author

Buvananand Vendotha
