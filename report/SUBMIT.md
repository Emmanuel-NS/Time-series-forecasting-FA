# What you still need to do

Everything else for this assignment is finished and pushed to
https://github.com/Emmanuel-NS/Time-series-forecasting-FA

Only these steps need **you**:

## 1. Record the video (7–10 minutes)

Use `report/VIDEO_GUIDE.md`. Screen-record with the repo open. Upload the video
wherever your course asks (Drive, YouTube unlisted, Canvas, etc.).

## 2. Put the video URL into the report

Open `report/report.md`, find reference **[34]** near the end, and replace the
placeholder with your video link. Example:

```text
[34] Video presentation: https://youtu.be/YOUR-VIDEO-ID
```

Then rebuild and push (PowerShell, from the project folder):

```powershell
cd 'C:\Users\PC\Ml tickines1 FA'
python scripts/06_build_report.py
git -c user.name='Emmanuel NSABAGASANI' -c user.email='nsabagasaniemm3@gmail.com' add report/report.md report/report.pdf
git -c user.name='Emmanuel NSABAGASANI' -c user.email='nsabagasaniemm3@gmail.com' commit -m "Add video presentation URL to report references"
git push
```

## 3. Submit the PDF

Upload `report/report.pdf` as the assignment file.

---

**Optional but smart before the viva:** read Section 8 once and tweak any wording
that does not match how you actually used the assistant. Then skim
`report/VIDEO_GUIDE.md` → “The hardest questions you are likely to get”.
