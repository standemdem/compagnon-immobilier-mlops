const map = L.map("map").setView(
    [48.8566, 2.3522],
    12
);

L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        attribution:
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }
).addTo(map);

let marker = null;
let arrondissementsData = null;
let selectedLatitude = null;
let selectedLongitude = null;
let selectedArrondissement = null;

const latitudeElement =
    document.getElementById("latitude");

const longitudeElement =
    document.getElementById("longitude");

const arrondissementElement =
    document.getElementById("arrondissement");


fetch("data/arrondissements.geojson")
    .then(response => {
        if (!response.ok) {
            throw new Error(
                "Impossible de charger les arrondissements."
            );
        }

        return response.json();
    })
    .then(data => {
        arrondissementsData = data;

        L.geoJSON(data, {
            style: {
                color: "#4b5563",
                weight: 1,
                fillOpacity: 0.05
            }
        }).addTo(map);
    })
    .catch(error => {
        console.error(error);

        arrondissementElement.textContent =
            "Erreur de chargement";
    });


function formatArrondissement(number) {
    if (number === 1) {
        return "Paris 1er Arrondissement";
    }

    return `Paris ${number}e Arrondissement`;
}


function findArrondissement(latitude, longitude) {

    if (arrondissementsData === null) {
        return null;
    }

    const point = turf.point([
        longitude,
        latitude
    ]);

    for (const feature of arrondissementsData.features) {

        if (
            turf.booleanPointInPolygon(
                point,
                feature
            )
        ) {
            const number =
                Number(feature.properties.c_ar);

            return formatArrondissement(number);
        }
    }

    return null;
}


map.on("click", function (event) {

    const latitude = event.latlng.lat;
    const longitude = event.latlng.lng;

    if (marker !== null) {
        map.removeLayer(marker);
    }

    marker = L.marker([
        latitude,
        longitude
    ]).addTo(map);

    latitudeElement.textContent =
        latitude.toFixed(6);

    longitudeElement.textContent =
        longitude.toFixed(6);

    const arrondissement =
        findArrondissement(
            latitude,
            longitude
        );

    if (arrondissementsData === null) {

        arrondissementElement.textContent =
            "Chargement des arrondissements...";

    } else if (arrondissement !== null) {

        arrondissementElement.textContent =
            arrondissement;

    } else {

        arrondissementElement.textContent =
            "Hors de Paris";
    }

    selectedLatitude = latitude;
    selectedLongitude = longitude;
    selectedArrondissement = arrondissement;
});

const form = document.getElementById("prediction-form");
const resultElement = document.getElementById("result");
const predictionValueElement =
    document.getElementById("prediction-value");

const errorMessageElement =
    document.getElementById("error-message");


form.addEventListener("submit", async function (event) {
    event.preventDefault();

    resultElement.classList.add("hidden");
    errorMessageElement.classList.add("hidden");

    if (
        selectedLatitude === null ||
        selectedLongitude === null ||
        selectedArrondissement === null
    ) {
        errorMessageElement.textContent =
            "Veuillez sélectionner un emplacement dans Paris.";

        errorMessageElement.classList.remove("hidden");
        return;
    }

    const surface =
        Number(document.getElementById("surface").value);

    const pieces =
        Number(document.getElementById("pieces").value);

    const hasDependance =
        document.getElementById("dependance").value === "true";

    const payload = {
        scope: "paris",
        surface_reelle_bati: surface,
        nombre_pieces_principales: pieces,
        latitude: selectedLatitude,
        longitude: selectedLongitude,
        has_dependance: hasDependance,
        nom_commune: selectedArrondissement
    };

    try {
        const response = await fetch("/demo/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(
                `Erreur API : ${response.status}`
            );
        }

        const data = await response.json();

        predictionValueElement.textContent =
            Math.round(data.prix_m2)
                .toLocaleString("fr-FR");

        resultElement.classList.remove("hidden");

    } catch (error) {
        console.error(error);

        errorMessageElement.textContent =
            "Impossible d'obtenir une estimation.";

        errorMessageElement.classList.remove("hidden");
    }
});