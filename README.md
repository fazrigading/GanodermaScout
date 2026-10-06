# GanodermaScout

GanodermaScout helps plantation staff inspect oil palms for Ganoderma boninense. A user uploads a photo of a palm base and asks a question. The system detects fruiting bodies, labels each as a primordium or a mature basidiocarp, and returns management advice that cites its sources. It stores each inspection by palm and block, so it can show how infection changes over time and alert an agronomist when a block crosses a threshold.

The repo also contains an evaluation harness. It compares four detectors (YOLOv12, YOLOv13, RT-DETRv3, RF-DETR), three retrieval modes (dense, hybrid, hybrid with reranker), and memory on versus off, and publishes the results on a benchmark page.

A detection is decision support and not a diagnosis. Fruiting bodies are generally a late sign, so the system never reports a palm as healthy because nothing was detected.
