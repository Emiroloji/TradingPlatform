# Hisse evreni

## bist100.csv

- Oluşturma tarihi: 2026-09-26
- Yöntem: Bilinen BIST 100 / BIST 100 adayı hisseler yfinance üzerinde doğrulandı (son 10 günde veri var mı);
  geçerli 111 sembolden son 3 ayın ortalama TL işlem hacmine göre ilk 100 seçildi.
  Resmi Borsa İstanbul listesi kazınmadı (KURALLAR §10).
- **Bu liste resmi BIST 100 bileşimi değildir, vekildir.** Resmi dönemsel liste elle alınıp bu dosya güncellenmelidir.
- Sektörler, KURALLAR §7 sektör limiti için elle atanmıştır.

## Survivorship bias (KURALLAR §3)

Geçmiş endeks üyeliği verisi **yoktur**. Liste bugünkü (2026-09) hayatta olan ve likit hisselerden oluşur;
geçmişte endeksten çıkan / işlem görmeyen hisseler dahil değildir. Bu nedenle backtest sonuçları
olduğundan iyi görünebilir. Bu kısıt her backtest raporunda açıkça yazılacaktır.

## Endeks

XU100 benchmark sembolü config.yaml'dan (`markets.bist.benchmark: XU100.IS`) okunur; benchmarks.csv bilgi amaçlıdır.

## sp500.csv

- Oluşturma tarihi: 2026-09-26
- Kaynak: Wikipedia "List of S&P 500 companies" (constituents tablosu; CC BY-SA, programatik okumaya izinli).
- Kolonlar: ticker (yfinance biçimi, "BRK.B" → "BRK-B"), name, sector (GICS), cik (SEC EDGAR kimliği).
- Survivorship bias BIST ile aynı: bugünkü bileşim; geçmiş üyelik verisi yok.
