# bottle3d

PDF-чертёж стеклянной тары → 3D-модель для Blender (вращение реального контура из векторного PDF)
и превью-рендер для рекламы. План: [docs/PLAN.md](docs/PLAN.md).

Сейчас готово ядро и CLI (M0 + большая часть M1). Десктопное приложение — следующий шаг (M2).

## Установка (Windows)

1. [uv](https://docs.astral.sh/uv/) и [Blender 4.2 LTS](https://www.blender.org/download/lts/4-2/)
   (обычная установка в `C:\Program Files\Blender Foundation\` находится автоматически,
   иначе — `--blender путь\к\blender.exe` или переменная `BLENDER`).
2. `uv sync`

## Запуск

```powershell
uv run bottle3d run чертёж.pdf --job job.yaml --profile profiles/panevezio.yaml --out out
```

В `out/` появятся:

- `debug.png` — чертёж с найденной осью (зелёная), рамкой вида (синяя) и контуром (красный).
  **Смотреть первым делом.**
- `report.md` / `report.json` — сверка «по чертежу → в модели», толщина стенки, объёмы, статус
  `ok` / `needs_review` и что именно требует внимания.
- `profile.json` — сечение для Blender (метры).
- `model.blend`, `preview.png` (и `model.glb` с `--glb`).
- `job.yaml` — все настройки этого прогона, включая угаданные автоматически. Поправьте в нём
  ось, рамку вида или толщину линии контура и запустите снова с `--job out/job.yaml`.

`--no-blender` останавливается после `profile.json`; `--engine BLENDER_EEVEE_NEXT` — быстрый
превью-рендер на видеокарте.

## job.yaml

```yaml
axis_x_mm: 117.2              # ось главного вида, мм от левого края листа (пусто = искать)
view_bbox_mm: [37, 60, 197, 350]  # x0, y0, x1, y1 главного вида, мм, Y снизу (пусто = искать)
contour_width_pt: 0.84        # толщина линий контура (пусто = самая толстая из частых)
base_thickness_mm: 10         # толщина дна: из штампа не определяется, задаётся явно
spec:
  name: Whisky 700 Cork
  overall_height_mm: 246.0    # опорные размеры для калибровки масштаба
  body_diameter_mm: 88.3
  base_diameter_mm: 73.0
  glass_weight_g: 500         # → толщина стенки
  brimful_capacity_ml: 720    # → проверка внешнего объёма
  nominal_capacity_ml: 700    # вместе с fill_height_mm → проверка внутреннего профиля
  fill_height_mm: 53
  neck_inner_diameter_mm: 18.7
  glass_colour: Flint
```

## Разработка

```powershell
uv run pytest        # тест с Blender пропускается, если blender не найден
uv run ruff check .
```

Тесты используют синтетические PDF (`tests/synthetic/make_pdf.py`) с известной геометрией —
это точная истина для контура, калибровки и объёмов.
