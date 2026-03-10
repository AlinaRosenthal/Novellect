import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


class FineTuner:
    """
    Класс для дообучения модели на литературных текстах
    """

    def __init__(self, base_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.base_model_name = base_model_name
        self.model = None
        self.examples = []

    def load_training_data(self, vector_db_file='vector_db.npz'):
        """
        Загружает данные для обучения из проиндексированных книг
        """
        print("[FINE-TUNE] Загрузка данных для обучения...")

        # Загружаем векторную БД
        if not os.path.exists(vector_db_file):
            print("[ERROR] Векторная БД не найдена")
            return False

        try:
            data = np.load(vector_db_file, allow_pickle=True)
            metadata = json.loads(data['metadata'].item()) if data['metadata'].size > 0 else []

            if len(metadata) < 10:
                print("[ERROR] Недостаточно данных для обучения (нужно минимум 10 чанков)")
                return False

            print(f"[FINE-TUNE] Загружено {len(metadata)} чанков")

            # Создаем позитивные и негативные пары
            self._create_training_pairs(metadata)
            return True

        except Exception as e:
            print(f"[ERROR] Ошибка загрузки данных: {e}")
            return False

    def _create_training_pairs(self, metadata, num_pairs=100):
        """
        Создает обучающие пары
        """
        print(f"[FINE-TUNE] Создание {num_pairs} обучающих пар...")

        # Группируем чанки по книгам
        books = {}
        for i, m in enumerate(metadata):
            book_id = m.get('book_id', f'unknown_{i}')
            if book_id not in books:
                books[book_id] = []
            books[book_id].append(m)

        # Создаем пары
        import random
        examples = []
        book_ids = list(books.keys())

        if len(book_ids) < 2:
            print("[ERROR] Нужно минимум 2 книги для создания пар")
            return

        for _ in range(min(num_pairs, 500)):
            try:
                # Выбираем случайную книгу
                book_id = random.choice(book_ids)
                book_chunks = books[book_id]

                if len(book_chunks) < 2:
                    continue

                # Выбираем два разных чанка из одной книги
                pos_idx1, pos_idx2 = random.sample(range(len(book_chunks)), 2)
                pos_chunk1 = book_chunks[pos_idx1]
                pos_chunk2 = book_chunks[pos_idx2]

                # Создаем запрос из первого чанка
                query = self._create_query_from_chunk(pos_chunk1.get('chunk', ''))

                # Позитивный пример - второй чанк
                positive = pos_chunk2.get('chunk', '')

                # Негативный пример - из другой книги
                other_book_id = random.choice([b for b in book_ids if b != book_id])
                negative = random.choice(books[other_book_id]).get('chunk', '')

                if query and positive and negative:
                    examples.append(InputExample(texts=[query, positive, negative]))

            except Exception as e:
                continue

        self.examples = examples
        print(f"[FINE-TUNE] Создано {len(examples)} triplet-примеров")

    def _create_query_from_chunk(self, chunk, max_words=20):
        """Создает запрос из начала чанка"""
        if not chunk:
            return ""
        words = chunk.split()[:max_words]
        return ' '.join(words)

    def fine_tune(self, epochs=3, batch_size=4, learning_rate=2e-5, output_path='fine_tuned_literary_model'):
        """
        Запускает дообучение модели
        """
        if not self.examples:
            print("[ERROR] Нет данных для обучения")
            return False

        print(f"[FINE-TUNE] Начало дообучения модели...")
        print(f"  - Эпох: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Примеров: {len(self.examples)}")

        try:
            # Загружаем базовую модель
            self.model = SentenceTransformer(self.base_model_name)

            # Создаем DataLoader
            train_dataloader = DataLoader(self.examples, shuffle=True, batch_size=batch_size)

            # Используем MultipleNegativesRankingLoss
            train_loss = losses.MultipleNegativesRankingLoss(self.model)

            # Дообучаем
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=100,
                optimizer_params={'lr': learning_rate},
                output_path=output_path,
                show_progress_bar=True,
                save_best_model=True
            )

            print(f"[FINE-TUNE] ✅ Модель успешно дообучена и сохранена в {output_path}")

            # Сохраняем метаданные
            os.makedirs(output_path, exist_ok=True)
            metadata = {
                'base_model': self.base_model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_examples': len(self.examples),
                'output_path': output_path
            }

            with open(os.path.join(output_path, 'training_metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"[ERROR] Ошибка дообучения: {e}")
            return False


def fine_tune_model(epochs=3, batch_size=4):
    """
    Удобная функция для дообучения модели
    """
    tuner = FineTuner()
    if tuner.load_training_data():
        return tuner.fine_tune(epochs=epochs, batch_size=batch_size)
    return False


if __name__ == "__main__":
    # Тестовый запуск
    fine_tune_model(epochs=1, batch_size=2)