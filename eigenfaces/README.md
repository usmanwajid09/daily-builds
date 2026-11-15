# Eigenfaces

Face recognition from scratch: PCA on face images gives the
"eigenfaces"; a new face is recognised by projecting it into eigenface
space and finding the nearest training face. Reconstruction from a few
eigenfaces shows how much information the components keep.

## Why

Eigenfaces (Turk & Pentland, 1991) is the classic PCA approach to face
recognition, and it is a great way to see PCA working on real-shaped
data: faces live near a low-dimensional subspace, so a handful of
eigenfaces captures most of the variation.

## Usage

```bash
pip install -r requirements.txt
python eigenfaces/demo.py
```

This builds 3 synthetic identities (smooth random blob patterns plus
noise), fits 8 eigenfaces, and recognises a fresh noisy copy of one
identity.

Example output:

```
Training faces: 15 (3 identities), image size: (16, 16)
Eigenfaces kept: 8, variance explained: 0.986
Training recognition accuracy: 1.00
Query face (true identity 1) recognised as: 1
Mean reconstruction MSE: 0.00080 (max 0.00122)
```

## Run tests

```bash
python -m pytest tests/test_eigenfaces.py -q
```
