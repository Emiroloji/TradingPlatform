# Temkinli Hisse Tarama ve Sinyal Sistemi

BIST 100 ve S&P 500 hisselerini günlük tarayan, kurumsal toplama işaretlerini çoklu indikatör,
walk-forward makine öğrenmesi ve Gemini haber/risk kontrolüyle değerlendiren **karar destek** sistemi.
Otomatik emir vermez; tüm çıktılar geçmiş veriye dayalı olasılıklardır. **Yatırım tavsiyesi değildir.**

Kurallar, mimari ve fazlar proje notebook'undadır (KURALLAR / MİMARİ / FAZLAR / PROJE).

## Kurulum
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```
macOS'ta LightGBM için `brew install libomp`. Sunucu kurulumu: [deploy/README.md](deploy/README.md).

## Komutlar
```bash
python main.py fetch           # veri çek, temizle, kalite raporu
python main.py features        # özellik ve skor tablosu
python main.py backtest        # kural bazlı baseline (+ kurulum OOS özeti)
python main.py train           # walk-forward LightGBM
python main.py backtest --ml   # ML vs baseline, canlı onay kararı
python main.py run             # günlük tam akış → rapor + Telegram
python main.py schedule        # hafta içi otomatik çalışma
streamlit run report/dashboard.py
pytest
```

## Klasörler
`data/` veri · `features/` özellikler · `backtest/` motor ve metrikler · `model/` etiket, eğitim, tahmin ·
`ai/` haber, Gemini, yorum · `signals/` filtre, risk, günlük · `report/` Telegram ve panel · `deploy/` sunucu ·
`storage/` çalışma verisi (git dışı)
