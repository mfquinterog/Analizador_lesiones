// Función para simular un clic en un elemento
function simulateClick(tabID) {
    document.getElementById(tabID).click();
}

// Función para predecir automáticamente al cargar la página
function predictOnLoad() {
    setTimeout(simulateClick.bind(null, 'predict-button'), 500);
}

// Manejo del cambio de imagen seleccionada
$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    };

    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);

    // Simular un clic en el botón de predicción después de un retardo
    setTimeout(simulateClick.bind(null, 'predict-button'), 500);
});

// Función para agregar una capa de entrada si el modelo no tiene definida una
const ensureInputLayer = (model) => {
    if (!model.inputs || model.inputs.length === 0) {
        console.log("Agregando capa de entrada al modelo...");
        const input = tf.input({ shape: [224, 224, 3] });
        const output = model.apply(input);
        return tf.model({ inputs: input, outputs: output });
    }
    return model;
};

// Variable para almacenar el modelo cargado
let model;

// Cargar el modelo y manejar errores
(async function () {
    $('.progress-bar').show();

    try {
        // Cargar modelo
        const baseModel = await tf.loadLayersModel('model_kerasnative_v4/model.json');
        console.log("Modelo base cargado exitosamente");

        // Asegurar que el modelo tenga una capa de entrada
        model = ensureInputLayer(baseModel);

        console.log("Modelo final listo para predicciones");

        // Establecer imagen por defecto
        $("#selected-image").attr("src", "assets/samplepic.jpg");

        // Ocultar spinner de carga del modelo
        $('.progress-bar').hide();

        // Predecir automáticamente al cargar la página
        predictOnLoad();
    } catch (error) {
        console.error("Error al cargar el modelo:", error);
        alert("Hubo un problema al cargar el modelo. Verifica los archivos del modelo.");
        $('.progress-bar').hide();
    }
})();

// Manejo del clic en el botón de predicción
$("#predict-button").click(async function () {
    let image = $('#selected-image').get(0);

    // Preprocesar la imagen
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat();

    // Normalizar los datos de la imagen entre -1 y 1
    let offset = tf.scalar(127.5);
    tensor = tensor.sub(offset).div(offset).expandDims();

    // Realizar predicción con el modelo cargado
    try {
        let predictions = await model.predict(tensor).data();
        let top6 = Array.from(predictions)
            .map(function (p, i) {
                return {
                    probability: p,
                    className: TARGET_CLASSES[i] // Aquí asegúrate de que TARGET_CLASSES esté definido
                };
            })
            .sort(function (a, b) {
                return b.probability - a.probability;
            })
            .slice(0, 6);

        // Mostrar predicciones en la lista
        $("#prediction-list").empty();
        top6.forEach(function (p) {
            $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
        });
    } catch (error) {
        console.error("Error durante la predicción:", error);
        alert("Hubo un problema al realizar la predicción.");
    }
});
