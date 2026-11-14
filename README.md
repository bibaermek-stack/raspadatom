# raspadatom — атомдық спектрлерді тану ЖИ

1D-CNN + Transformer (Attention) моделі NIST ASD сызықтарынан жасалған синтетикалық спектрлерді оқып,
(1) қоспадағы түрлерді (multi-label) және (2) плазма температурасын болжайды.

```bash
pip install -r requirements.txt
python -m spectra_ai.nist            # NIST-тен data/nist_lines.csv жаңарту (дайын файл бар)
python -m spectra_ai.train --epochs 15
pytest
```

Қозғалтқыш: `spectra_ai/synth.py` — Больцман интенсивтілігі, pseudo-Voigt профилі, ығыс, базалық сызық, шу.
Нәтиже (15 эпоха, CPU, синтетикалық валидация): F1 0.855, precision 0.95, recall 0.78, T-MAE ≈ 690 K.
Түрлер: 29 (H I … O I). Нақты спектрометр деректерімен тексерілмеген.
