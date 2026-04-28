# Финальный отчет по ML-проекту: прогнозирование цены Airbnb

## 1. Название проекта
Прогнозирование цены объявления Airbnb по характеристикам объекта, хоста, отзывов и доступности.

## 2. Бизнес-задача
Цель проекта — построить модель, которая оценивает цену аренды жилья по признакам объявления Airbnb. Такая модель может помочь в задачах ценообразования, мониторинга рынка, аналитики конкурентов и поддержки принятия решений для хостов или платформы.

## 3. Общая информация о данных
- Источник данных: `rent_predictions/airbnb_train_fe_15000.csv`
- Количество строк: 15000
- Количество столбцов: 36
- Предметная область: Это датасет объявлений Airbnb для оценки аренды жилья.
- Смысл строки: Одна строка, вероятно, описывает одно объявление/объект размещения Airbnb с характеристиками жилья, хоста, отзывов, доступности и ценой.
- Пропуски до подготовки: 31176
- Дубликаты: 0

## 4. Целевая колонка и тип ML-задачи
- Целевая колонка: `price`
- Тип задачи: regression

## 5. Группы признаков из схемы
- **numeric**: `accommodates`, `bathrooms`, `bedrooms`, `beds`, `host_listings_count`, `host_total_listings_count`, `latitude`, `longitude`, `availability_30`, `availability_60`, `availability_90`, `availability_365`, `number_of_reviews`, `number_of_reviews_ltm`, `number_of_reviews_l30d`, `review_scores_rating`, `review_scores_cleanliness`, `review_scores_location`, `review_scores_value`, `reviews_per_month`
- **categorical**: not available
- **text**: `description`, `amenities`, `property_type`, `room_type`, `host_since`, `host_location`, `host_response_time`, `host_response_rate`, `host_acceptance_rate`, `host_is_superhost`, `city`, `has_availability`, `first_review`, `last_review`
- **datetime**: not available
- **boolean_like**: not available
- **id**: `id`
- **target**: `price`

## 6. Обзор качества данных
- До подготовки в данных было 31176 пропусков.
- Дубликаты строк отсутствуют.
- После подготовки итоговый датасет не содержит пропусков.
- Итоговая размерность подготовленного датасета: 15000 строк и 35 столбцов.

## 7. Шаги по подготовке данных
- Приведение типов признаков.
- Создание новых признаков на основе текста, удобств, соотношений между числовыми характеристиками, доступности и отзывов.
- Очистка колонок.
- Обработка пропусков.
- Кодирование и масштабирование признаков.
- Финальная сборка подготовленного датасета.

## 8. Созданные признаки
- `description_char_len` из `description` — Длина описания может отражать уровень детализации объявления и качество информации для модели.
- `description_word_count` из `description` — Число слов в описании дает простой сигнал о полноте текстового описания.
- `amenities_char_len` из `amenities` — Длина списка удобств может быть прокси для наполненности объекта.
- `amenities_item_count` из `amenities` — Количество удобств по разделителям отражает оснащенность жилья.
- `bathrooms_per_bedroom` из `bathrooms, bedrooms` — Соотношение ванных комнат к спальням может отражать уровень комфорта и планировку.
- `accommodates_per_bedroom` из `accommodates, bedrooms` — Число гостей на спальню показывает плотность размещения.
- `beds_per_bedroom` из `beds, bedrooms` — Отношение кроватей к спальням помогает оценить вместимость и удобство объекта.
- `availability_30_share` из `availability_30, availability_365` — Краткосрочная доступность относительно годовой показывает режим занятости объекта.
- `review_scores_avg_quality` из `review_scores_rating, review_scores_cleanliness` — Средняя оценка по качеству и общему рейтингу сглаживает шум отдельных метрик отзывов.
- `reviews_activity_ratio` из `number_of_reviews, reviews_per_month` — Соотношение общего числа отзывов и их частоты помогает оценить историю активности объекта.

## 9. Модели, которые были протестированы
- Ridge
- RandomForestRegressor

## 10. Информация о лучшей модели
- Лучшая модель: **RandomForestRegressor**
- Путь к модели: `artifacts/modeling/best_model.joblib`
- В качестве итоговой модели выбран RandomForestRegressor, так как он показал лучшие результаты среди протестированных моделей.

## 11. Лучшие метрики
- mae: 79.29192956019443
- rmse: 127.2761760276829
- r2: 0.6032765656314466

## 12. Настройка гиперпараметров
- Настройка гиперпараметров использовалась.
- Лучшие параметры:
  - max_depth: None
  - min_samples_leaf: 2
  - min_samples_split: 2
  - n_estimators: 200
- После настройки качество улучшилось относительно базовой версии RandomForestRegressor.

## 13. Сравнение с предыдущим запуском
Сравнение выполнено по общим метрикам:
- mae: было 79.29192956019443, стало 79.29192956019443 — без изменений
- r2: было 0.6032765656314466, стало 0.6032765656314466 — без изменений
- rmse: было 127.2761760276829, стало 127.2761760276829 — без изменений

## 14. Бизнес-интерпретация результатов
Модель RandomForestRegressor объясняет заметную часть вариативности цены и показывает ошибку в пределах, пригодных для аналитических сценариев, где нужна не точная котировка, а ориентир для ценообразования и сравнения объектов.
С точки зрения бизнеса это означает, что модель можно использовать как вспомогательный инструмент для оценки рыночной цены и выявления объектов, которые выглядят переоцененными или недооцененными относительно их характеристик.
При этом ошибки модели все еще существенны, поэтому результат следует применять как рекомендательный, а не как единственный источник истины.

## 15. Конечные артефакты
- `artifacts/data_description/data_description_summary.json`
- `artifacts/data_description/eda_artifacts.json`
- `artifacts/data_preparation/data_preparation_summary.json`
- `artifacts/data_preparation/prepared_dataset.csv`
- `artifacts/modeling/final_metrics.json`
- `artifacts/modeling/best_model.joblib`
- `artifacts/modeling/tuned_models/tuned_RandomForestRegressor.joblib`
- `artifacts/reporting/final_report.md`
- `artifacts/reporting/reporting_summary.json`
