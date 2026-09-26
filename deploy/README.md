# Sunucuya kurulum (Linux + systemd)

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

## 3. Veri ve model (ilk kurulum)
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
```

## 4. Servisler
`deploy/*.service` içindeki `KULLANICI` ve `/opt/borsa` değerlerini düzenle:
```bash
sudo cp deploy/borsa-scheduler.service deploy/borsa-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now borsa-scheduler borsa-dashboard
systemctl status borsa-scheduler            # "Scheduled bist: weekdays 18:45 Europe/Istanbul"
journalctl -u borsa-scheduler -f            # canlı log
```
Zamanlayıcı saat dilimini config'ten alır (Europe/Istanbul); sunucunun saat dilimi önemli değildir.

## 5. Panel erişimi
Panel yalnızca `127.0.0.1:8501`'e bağlıdır. Kendi bilgisayarından:
```bash
ssh -L 8501:127.0.0.1:8501 KULLANICI@SUNUCU
# tarayıcı: http://127.0.0.1:8501
```
Kalıcı web erişimi istenirse önüne HTTPS + şifreli (basic auth) nginx konmalı; paneli doğrudan 0.0.0.0'a açma.

## 6. Faz 5 kabul kriteri
Sistem 2 hafta boyunca elle müdahalesiz çalışmalı ve her iş günü Telegram'a rapor gelmeli.
Kontrol: `storage/reports/daily_bist_*.md` her iş günü için var mı, `storage/logs/borsa.log`'da ERROR var mı.
