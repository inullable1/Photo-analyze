const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const loading = document.querySelector(".loading");
let currentPredictions = [];

// Обработка drag & drop
dropZone.addEventListener("dragover", (e) => {
	e.preventDefault();
	dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
	dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
	e.preventDefault();
	dropZone.classList.remove("dragover");
	const file = e.dataTransfer.files[0];
	if (file) handleFile(file);
});

dropZone.addEventListener("click", () => {
	fileInput.click();
});

fileInput.addEventListener("change", (e) => {
	const file = e.target.files[0];
	if (file) handleFile(file);
});

function handleFile(file) {
	if (!file.type.startsWith("image/")) {
		alert("Пожалуйста, загрузите изображение");
		return;
	}

	const formData = new FormData();
	formData.append("file", file);

	loading.style.display = "flex";

	fetch("/predict", {
		method: "POST",
		body: formData,
	})
		.then((response) => response.json())
		.then((data) => {
			if (data.error) {
				throw new Error(data.error);
			}

			// Сохраняем предсказания для использования при клике
			currentPredictions = data.predictions;

			// Отображаем оба изображения
			const originalImage = document.getElementById("originalImage");
			const markedImage = document.getElementById("markedImage");

			originalImage.src = data.original_image_url;
			markedImage.src = data.marked_image_url;

			originalImage.style.display = "block";
			markedImage.style.display = "block";

			// Синхронизируем размеры изображений
			originalImage.onload = () => {
				const markedContainer = markedImage.parentElement;
				markedContainer.style.width =
					originalImage.parentElement.offsetWidth + "px";
			};

			// Очищаем предыдущие рамки
			clearBoundingBoxes();

			// Отображаем предсказания
			displayPredictions(data.predictions);

			// Отображаем графики
			if (data.graph_url) {
				const graphsImage = document.getElementById("graphsImage");
				graphsImage.src = data.graph_url;
				graphsImage.style.display = "block";
			}
		})
		.catch((error) => {
			console.error("Error:", error);
			alert("Произошла ошибка при обработке изображения");
		})
		.finally(() => {
			loading.style.display = "none";
		});
}

function displayPredictions(predictions) {
	const predictionsContainer = document.getElementById("predictions");
	predictionsContainer.innerHTML = "";

	predictions.forEach((pred, index) => {
		const card = document.createElement("div");
		card.className = "col-md-6";
		card.innerHTML = `
            <div class="card prediction-card" data-index="${index}">
                <div class="card-body">
                    <h6 class="card-title">${pred.class}</h6>
                    <p class="card-text">
                        Уверенность: ${pred.confidence}%<br>
                        Координаты: (${pred.coordinates.x1}, ${pred.coordinates.y1}) - 
                        (${pred.coordinates.x2}, ${pred.coordinates.y2})
                    </p>
                </div>
            </div>
        `;

		// Добавляем обработчик клика
		card.querySelector(".prediction-card").addEventListener("click", () => {
			// Убираем активный класс у всех карточек
			document
				.querySelectorAll(".prediction-card")
				.forEach((c) => c.classList.remove("active"));
			// Добавляем активный класс к выбранной карточке
			card.querySelector(".prediction-card").classList.add("active");
			// Показываем рамку для выбранного объекта
			showBoundingBox(pred.coordinates, index);
		});

		predictionsContainer.appendChild(card);
	});
}

function showBoundingBox(coords, index) {
	// Очищаем предыдущие рамки
	clearBoundingBoxes();

	const imageContainer = document.querySelector(".image-container");
	const image = document.getElementById("originalImage");

	// Создаем рамку
	const box = document.createElement("div");
	box.className = "bounding-box active";

	// Вычисляем размеры и позицию рамки относительно изображения
	const imageRect = image.getBoundingClientRect();
	const scaleX = imageRect.width / image.naturalWidth;
	const scaleY = imageRect.height / image.naturalHeight;

	box.style.left = `${coords.x1 * scaleX}px`;
	box.style.top = `${coords.y1 * scaleY}px`;
	box.style.width = `${(coords.x2 - coords.x1) * scaleX}px`;
	box.style.height = `${(coords.y2 - coords.y1) * scaleY}px`;

	// Добавляем подпись с классом и уверенностью
	const label = document.createElement("div");
	label.style.position = "absolute";
	label.style.top = "-25px";
	label.style.left = "0";
	label.style.backgroundColor = "#0d6efd";
	label.style.color = "white";
	label.style.padding = "2px 6px";
	label.style.borderRadius = "3px";
	label.style.fontSize = "12px";
	label.textContent = `${currentPredictions[index].class} (${currentPredictions[index].confidence}%)`;
	box.appendChild(label);

	imageContainer.appendChild(box);
}

function clearBoundingBoxes() {
	const boxes = document.querySelectorAll(".bounding-box");
	boxes.forEach((box) => box.remove());
}

// Обработка клика по классам
document.querySelectorAll(".class-badge").forEach((badge) => {
	badge.addEventListener("click", () => {
		const className = badge.dataset.class;
		badge.classList.toggle("active");

		// Фильтруем предсказания по выбранному классу
		const filteredPredictions = currentPredictions.filter(
			(p) => p.class === className
		);
		displayPredictions(filteredPredictions);
	});
});
