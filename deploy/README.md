# Sunucuya kurulum

## A. Dokploy (önerilen)

Depodaki `Dockerfile` + `docker-compose.yml` üç servis kurar: **borsa-db** (TimescaleDB, dışarı kapalı),
**borsa-scheduler** (ilk açılışta `bootstrap`, sonra hafta içi günlük akış + aylık yeniden eğitim),
**borsa-dashboard** (Streamlit, şifreli). Veriler adlandırılmış volume'larda kalıcıdır (yeniden deploy'da silinmez).
Servis adları "borsa-" ön eklidir: sunucudaki diğer projelerle aynı Dokploy ağında "db" gibi ortak bir ad
yanlış konteynere çözümlenebilir.

1. Dokploy → **Projects** → yeni proje → **Create Service → Compose**.
2. **Provider:** GitHub → depo `Emiroloji/TradingPlatform` (özel depo: Dokploy GitHub uygulamasına erişim ver),
   branch `main`, **Compose Path** `./docker-compose.yml`.
3. **Environment** sekmesi (değerler depoya girmez):
   ```
   GEMINI_API_KEY=...
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   SEC_USER_AGENT=TradingPlatform ad@eposta.com
   POSTGRES_PASSWORD=uzun-rastgele-bir-şifre
   DASHBOARD_PASSWORD=panel-şifresi
   REDDIT_CLIENT_ID=            # isteğe bağlı
   REDDIT_CLIENT_SECRET=        # isteğe bağlı
   ```
4. **Deploy**. İlk açılışta scheduler bootstrap'i çalıştırır (~10 dk: BIST + S&P 500, SEC Form 4 ve 13F).
   Loglarda `Scheduled bist: weekdays 18:45` görünce hazırdır.
5. **Domains** sekmesi → servis `borsa-dashboard`, port `8501`, HTTPS açık → alan adın. Panel şifreyi sorar.
6. Kontrol: Telegram'a hafta içi 18:45'ten sonra BIST, 23:45'ten sonra ABD raporu gelir.

Kaynak ihtiyacı (Linux konteynerde ölçüldü, BIST + S&P 500): günlük akış tepe ~0,9 GB, aylık eğitim ve
bootstrap'in en ağır adımı ~1,5 GB, zamanlayıcı boşta ~260 MB; disk ~3 GB (veri + veritabanı).
Her zamanlanmış iş ve her bootstrap adımı ayrı alt süreçte çalışır, bittiğinde belleği işletim sistemine
döner. Compose'daki sınırlar: borsa-scheduler 1600 MB, borsa-db 512 MB, borsa-dashboard 512 MB.
Alt süreç beklenmedik şekilde ölürse (ör. bellek yetmezse) Telegram'a uyarı gider.
Yerelde doğrulandı: imaj derleniyor, bootstrap yereldeki sonuçları birebir üretiyor, 130 test konteynerde geçiyor.

---

## B. Elle kurulum (Linux + systemd)

## 1. Kod ve ortam
```bash
# projeyi /opt/borsa altına kopyala (git veya rsync); .env ve storage/ ayrıca taşınır
sudo apt install -y python3.13-venv libgomp1        # libgomp1: LightGBM için (Python 3.11+ yeterli)
cd /opt/borsa
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt           # birebir aynı sürümler için: -r deploy/requirements.lock
```

## 2. Gizli anahtarlar
`.env` dosyasını sunucuya ayrıca kopyala (git'e girmez), izinlerini kısıtla:
```bash
chmod 600 .env
```

## 3. Veritabanı (PostgreSQL + TimescaleDB, ücretsiz)
Parquet yerel çalışma katmanıdır; TimescaleDB kalıcı analitik veritabanıdır (her günlük akış sonunda senkronize edilir).
`.env`: `POSTGRES_PASSWORD` ve `DATABASE_URL=postgresql://borsa:ŞİFRE@127.0.0.1:5432/borsa`.

- Docker ile: `docker compose --env-file .env -f deploy/docker-compose.yml up -d` (yalnızca 127.0.0.1)
- ya da paketle (Ubuntu): TimescaleDB'nin resmî apt deposundan `timescaledb-2-postgresql-16`, sonra
  `timescaledb-tune --quiet --yes`, `CREATE ROLE borsa LOGIN PASSWORD '...'; CREATE DATABASE borsa OWNER borsa;`
  ve `borsa` veritabanında `CREATE EXTENSION timescaledb;`

İlk aktarım: `.venv/bin/python main.py db-sync`. Veritabanına erişilemezse günlük akış durmaz, Telegram'a uyarı gider.

## 4. Veri ve model (ilk kurulum)
İki seçenek:
- **Taşı:** yereldeki `storage/` ve `model/registry/` klasörlerini sunucuya kopyala, ya da
- **Yeniden üret:**
```bash
.venv/bin/python main.py fetch
.venv/bin/python main.py features
.venv/bin/python main.py backtest        # kurulumun OOS özeti (setup_bist.json) — filtre bunu okur
.venv/bin/python main.py train
.venv/bin/python main.py backtest --ml   # modelin canlı onay kararı
.venv/bin/python main.py run             # elle bir tam akış; Telegram'a rapor gelmeli
.venv/bin/python main.py drift           # model kayması raporu
```

## 5. Servisler
`deploy/*.service` içindeki `KULLANICI` ve `/opt/borsa` değerlerini düzenle:
```bash
sudo cp deploy/borsa-scheduler.service deploy/borsa-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now borsa-scheduler borsa-dashboard
systemctl status borsa-scheduler            # "Scheduled bist: weekdays 18:45 Europe/Istanbul"
journalctl -u borsa-scheduler -f            # canlı log
```
Zamanlayıcı saat dilimini config'ten alır (Europe/Istanbul); sunucunun saat dilimi önemli değildir.

## 6. Panel erişimi
Panel yalnızca `127.0.0.1:8501`'e bağlıdır. Kendi bilgisayarından:
```bash
ssh -L 8501:127.0.0.1:8501 KULLANICI@SUNUCU
# tarayıcı: http://127.0.0.1:8501
```
Kalıcı web erişimi istenirse önüne HTTPS + şifreli (basic auth) nginx konmalı; paneli doğrudan 0.0.0.0'a açma.

## 7. Faz 5 kabul kriteri
Sistem 2 hafta boyunca elle müdahalesiz çalışmalı ve her iş günü Telegram'a rapor gelmeli.
Kontrol: `storage/reports/daily_bist_*.md` her iş günü için var mı, `storage/logs/borsa.log`'da ERROR var mı.
