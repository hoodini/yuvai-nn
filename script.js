// הגדרות תוכן קבועות - להעביר לתחילת הקובץ
const DICTIONARY_CONTENT = {
    'רשת נוירונים': {
        icon: 'fa-brain',
        content: 'מודל מתמטי המחקה את פעולת המוח האנושי. מורכב משכבות של נוירונים מלאכותיים המחוברים ביניהם.'
    },
    'משקולות': {
        icon: 'fa-weight-hanging',
        content: 'פרמטרים מתכווננים המייצגים את חוזק הקשרים בין הנוירונים. הרשת לומדת על ידי עדכון המשקולות.'
    },
    'פונקציית הפעלה': {
        icon: 'fa-wave-square',
        content: 'פונקציה מתמטית המוסיפה אי-לינאריות לרשת. דוגמאות נפוצות: ReLU, Sigmoid, Tanh.'
    },
    'Deep Learning': {
        icon: 'fa-brain',
        content: `שיטת למידה מתקדמת המבוססת על רשתות נוירונים עמוקות. בניגוד לשיטות למידת מכונה מסורתיות, Deep Learning מסוגל:
        • ללמוד מאפיינים מורכבים באופן אוטומטי
        • לעבד מידע גולמי (תמונות, טקסט, קול) ללא עיבוד מקדים
        • להשיג ביצועים מעולים במשימות מורכבות כמו ראייה ממוחשבת ועיבוד שפה טבעית`
    },
    'Machine Learning': {
        icon: 'fa-robot',
        content: `תחום שבו מכונות לומדות מדוגמאות במקום להיות מתוכנתות עם חוקים מפורשים. לדוגמה:
        • זיהוי ספאם: במקום לכתוב חוקים, המערכת לומדת מדוגמאות של מיילים
        • חיזוי מחירי דירות: לומדת מנתוני עבר במקום נוסחאות קבועות
        • המלצות תוכן: לומדת מהתנהגות משתמשים במקום חוקים קבועים`
    },
    'Structured vs Unstructured Data': {
        icon: 'fa-database',
        content: `שני סוגי מידע עיקריים בלמידת מכונה:
        
        Structured Data:
        • מאורגן בטבלאות עם שדות ברורים
        • לדוגמה: טבלת לקוחות עם גיל, הכנסה, מיקום
        • קל יחסית לעיבוד
        
        Unstructured Data:
        • מידע לא מאורגן כמו תמונות, טקסט חופשי, קול
        • מהווה 80% מהמידע בעולם
        • מאתגר יותר לעיבוד, Deep Learning מצטיין בזה`
    },
    'LLM & Gen AI': {
        icon: 'fa-magic',
        content: `Large Language Models & Generative AI:
        
        LLM:
        • מודלים ענקיים שמבינים ומייצרים טקסט
        • נוצרו מלמידה על טריליוני מילים
        • דוגמאות: GPT, Claude, BERT
        
        Generative AI:
        • מערכות שיוצרות תוכן חדש
        • יכולות ליצור טקסט, תמונות, קול, וידאו
        • דוגמאות: DALL-E, Stable Diffusion`
    },
    'עיבוד והכשרה': {
        icon: 'fa-microchip',
        content: `כלים ותשתיות לאימון מודלים:
        
        PyTorch & TensorFlow:
        • ספריות פופולריות לבניית רשתות נוירונים
        • PyTorch: גמיש ואינטואיטיבי, פופולרי במחקר
        • TensorFlow: תעשייתי ויציב, פופולרי בתעשייה
        
        CUDA & GPU:
        • CUDA: פלטפורמת NVIDIA לחישוב מקבילי
        • GPU: מעבד גרפי שמאיץ אימון פי 10-100 מ-CPU`
    },
    'אתגרים ופתרונות': {
        icon: 'fa-balance-scale',
        content: `אתגרים נפוצים באימון מודלים:
        
        Overfitting:
        • המודל "זוכר" את הדוגמאות במקום ללמוד
        • סימנים: ביצועים מעולים באימון, גרועים בטסט
        • פתרונות: Regularization, Dropout, יותר דאטה
        
        Underfitting:
        • המודל פשוט מדי/לא לומד מספיק
        • סימנים: ביצועים גרועים גם באימון
        • פתרונות: מודל מורכב יותר, אימון ארוך יותר
        
        Sweet Spot:
        • האיזון המושלם בין השניים
        • המודל מכליל היטב לדוגמאות חדשות
        • מושג על ידי ניטור מתמיד וכיוון היפר-פרמטרים`
    }
};

const METRICS_CONTENT = {
    'Accuracy': {
        icon: 'fa-bullseye',
        formula: '(TP + TN) / (TP + TN + FP + FN)',
        content: `דיוק - כמה פעמים המודל צדק מתוך סך כל התחזיות
        
        דוגמה פשוטה - זיהוי ספאם:
        • נבדקו 100 מיילים
        • המודל זיהה נכון 85 מיילים (45 ספאם אמיתי + 40 לא ספאם)
        • טעה ב-15 מיילים
        • Accuracy = 85/100 = 85%
        
        יתרונות: קל להבנה
        חסרונות: מטעה במקרים לא מאוזנים`,
        example: {
            data: [[90, 10], [20, 80]],
            calculation: 'Accuracy = (90 + 80) / (90 + 10 + 20 + 80) = 0.85 = 85%'
        }
    },
    'Precision': {
        icon: 'fa-crosshairs',
        formula: 'TP / (TP + FP)',
        content: `דיוק חיובי - מתוך מה שהמודל סיווג כחיובי, כמה באמת היה חיובי
        
        דוגמה פשוטה - זיהוי מחלות:
        • המודל אבחן 100 אנשים כחולים
        • מתוכם 80 באמת חולים
        • 20 היו בריאים (אזעקת שווא)
        • Precision = 80/100 = 80%
        
        חשיבות:
        • קריטי כשטעויות חיוביות יקרות
        • למשל: לא רוצים להטריד לקוחות בטעות`,
        example: {
            data: [[90, 10], [20, 80]],
            calculation: 'Precision = 90 / (90 + 20) = 0.82 = 82%'
        }
    },
    'Recall': {
        icon: 'fa-search',
        formula: 'TP / (TP + FN)',
        content: `רגישות - כמה מהמקרים החיוביים האמיתיים המודל הצליח למצוא
        
        דוגמה פשוטה - זיהוי הונאות:
        • היו 100 הונאות אמיתיות
        • המודל זיהה 70 מתוכן
        • פספס 30 הונאות
        • Recall = 70/100 = 70%
        
        חשיבות:
        • קריטי כשפספוס מקרים חיוביים מסוכן
        • למשל: אבחון מחלות, זיהוי הונאות`,
        example: {
            data: [[90, 10], [20, 80]],
            calculation: 'Recall = 90 / (90 + 10) = 0.90 = 90%'
        }
    }
};

// script.js
let hiddenLayerCount = 0;
let currentFramework = 'pytorch';
const canvas = document.getElementById('nnCanvas');
const ctx = canvas.getContext('2d');

// Layer Management Functions
function addHiddenLayer() {
    const container = document.getElementById('hiddenLayers');
    const div = document.createElement('div');
    div.className = 'layer-input';
    div.innerHTML = `
        <label><i class="fas fa-layer-group"></i> שכבה נסתרת ${hiddenLayerCount + 1}:</label>
        <input type="number" class="hidden-layer input" value="4" min="1">
        <button onclick="removeHiddenLayer(this)">
            <i class="fas fa-trash"></i>
        </button>
    `;
    container.appendChild(div);
    hiddenLayerCount++;
    updateVisualization();
}

function removeHiddenLayer(button) {
    button.parentElement.remove();
    hiddenLayerCount--;
    updateVisualization();
}

function getLayers() {
    const layers = [];
    layers.push(parseInt(document.getElementById('inputLayer').value));
    
    document.querySelectorAll('.hidden-layer').forEach(input => {
        layers.push(parseInt(input.value));
    });
    
    layers.push(parseInt(document.getElementById('outputLayer').value));
    return layers;
}

// Parameter Calculation Functions
function calculateParameters(layers) {
    let totalWeights = 0;
    let totalBiases = 0;
    
    for (let i = 0; i < layers.length - 1; i++) {
        const weights = layers[i] * layers[i + 1];
        const biases = layers[i + 1];
        totalWeights += weights;
        totalBiases += biases;
    }
    
    return {
        weights: totalWeights,
        biases: totalBiases,
        total: totalWeights + totalBiases
    };
}

function initializeClusters() {
    // יצירת 3 אשכולות עם נקודות רנדומליות
    const clusters = [
        { center: { x: 150, y: 150 }, color: '#9F7AEA', points: [] },
        { center: { x: 300, y: 150 }, color: '#4FD1C5', points: [] },
        { center: { x: 225, y: 300 }, color: '#F6AD55', points: [] }
    ];

    // יצירת נקודות רנדומליות סביב כל מרכז
    clusters.forEach(cluster => {
        for (let i = 0; i < 20; i++) {
            cluster.points.push({
                x: cluster.center.x + (Math.random() - 0.5) * 100,
                y: cluster.center.y + (Math.random() - 0.5) * 100,
                currentCluster: cluster
            });
        }
    });

    return clusters;
}

function updateClusterCenters() {
    demonstrationData.clusters.forEach(cluster => {
        if (cluster.points.length === 0) return;

        // חישוב ממוצע הנקודות
        const newCenter = cluster.points.reduce((acc, point) => ({
            x: acc.x + point.x,
            y: acc.y + point.y
        }), { x: 0, y: 0 });

        newCenter.x /= cluster.points.length;
        newCenter.y /= cluster.points.length;

        // עדכון הדרגתי של המרכז
        const progress = Math.min(demonstrationStep / 50, 1);
        cluster.center.x += (newCenter.x - cluster.center.x) * progress * 0.1;
        cluster.center.y += (newCenter.y - cluster.center.y) * progress * 0.1;
    });
}

function assignPointsToClusters() {
    // סידור מחדש של כל הנקודות
    const allPoints = demonstrationData.clusters.flatMap(c => c.points);
    
    // ניקוי הנקודות מכל האשכולות
    demonstrationData.clusters.forEach(cluster => {
        cluster.points = [];
    });

    // שיוך מחדש של כל נקודה לאשכול הקרוב ביותר
    allPoints.forEach(point => {
        let minDist = Infinity;
        let closestCluster = null;

        demonstrationData.clusters.forEach(cluster => {
            const dist = calculateDistance(point, cluster.center);
            if (dist < minDist) {
                minDist = dist;
                closestCluster = cluster;
            }
        });

        closestCluster.points.push(point);
        point.currentCluster = closestCluster;
    });
}

function calculateDistance(point1, point2) {
    const dx = point1.x - point2.x;
    const dy = point1.y - point2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function drawClusters(ctx) {
    // ציור הנקודות
    demonstrationData.clusters.forEach(cluster => {
        // ציור הנקודות
        cluster.points.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = cluster.color;
            ctx.fill();
        });

        // ציור מרכז האשכול
        ctx.beginPath();
        ctx.arc(cluster.center.x, cluster.center.y, 10, 0, Math.PI * 2);
        ctx.strokeStyle = cluster.color;
        ctx.lineWidth = 3;
        ctx.stroke();

        // ציור קווים מהמרכז לנקודות
        if (demonstrationStep % 10 < 5) { // הבהוב הקווים
            cluster.points.forEach(point => {
                ctx.beginPath();
                ctx.moveTo(cluster.center.x, cluster.center.y);
                ctx.lineTo(point.x, point.y);
                ctx.strokeStyle = `${cluster.color}40`; // צבע שקוף חלקית
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }
    });
}

function updateClusteringLoss() {
    if (!demonstrationData.lossHistory) {
        demonstrationData.lossHistory = [];
    }

    // חישוב ה-Loss הנוכחי (ממוצע המרחקים מהמרכזים)
    let totalDistance = 0;
    let totalPoints = 0;

    demonstrationData.clusters.forEach(cluster => {
        cluster.points.forEach(point => {
            totalDistance += calculateDistance(point, cluster.center);
            totalPoints++;
        });
    });

    const currentLoss = totalDistance / (totalPoints || 1);
    const normalizedLoss = Math.min(1, currentLoss / 100); // נרמול ל-0-1
    demonstrationData.lossHistory.push(normalizedLoss);

    // ציור גרף ה-Loss
    const lossCanvas = document.getElementById('lossCanvas');
    const ctx = lossCanvas.getContext('2d');
    
    // נקה את הקנבס
    ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    
    // ציור הגרף
    ctx.beginPath();
    ctx.strokeStyle = '#6B46C1';
    ctx.lineWidth = 2;

    demonstrationData.lossHistory.forEach((loss, i) => {
        const x = (i / 100) * lossCanvas.width;
        const y = lossCanvas.height - (loss * lossCanvas.height);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // הוספת תוויות
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.fillText('Loss', 10, 20);
    ctx.fillText('Epochs', lossCanvas.width - 50, lossCanvas.height - 10);
}

function updateStats() {
    const layers = getLayers();
    const params = calculateParameters(layers);
    
    document.getElementById('total-params').textContent = params.total.toLocaleString();
    document.getElementById('total-weights').textContent = params.weights.toLocaleString();
    document.getElementById('total-biases').textContent = params.biases.toLocaleString();
}

// Set up event listeners
document.getElementById('inputLayer').addEventListener('change', updateVisualization);
document.getElementById('outputLayer').addEventListener('change', updateVisualization);
document.addEventListener('change', function(e) {
    if (e.target.classList.contains('hidden-layer')) {
        updateVisualization();
    }
});

// Initial visualization
updateVisualization();

// Code Generation Functions
function generatePyTorchCode(layers) {
    return `import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # הגדרת השכבות
        ${layers.slice(0, -1).map((neurons, i) => 
            `self.layers.append(nn.Linear(${neurons}, ${layers[i + 1]}))`
        ).join('\n        ')}
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        return self.layers[-1](x)

# יצירת המודל
model = NeuralNetwork()

# הגדרת פונקציית Loss ואופטימייזר
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# אימון הרשת
def train(model, X, y):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`;
}

function generateTensorFlowCode(layers) {
    return `import tensorflow as tf
from tensorflow.keras import layers, models

# בניית המודל
model = models.Sequential()

# הוספת השכבות
${layers.slice(0, -1).map((neurons, i) => 
    i === 0 ? 
    `model.add(layers.Dense(${layers[i + 1]}, activation='relu', input_shape=(${neurons},)))` :
    `model.add(layers.Dense(${layers[i + 1]}, activation='relu'))`
).join('\n')}
model.add(layers.Dense(${layers[layers.length - 1]}))

# קומפילציה של המודל
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# אימון המודל
history = model.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    batch_size=32,
    validation_split=0.2
)`;
}

function updateCode(layers) {
    const codeOutput = document.getElementById('codeOutput');
    if (currentFramework === 'pytorch') {
        codeOutput.textContent = generatePyTorchCode(layers);
    } else {
        codeOutput.textContent = generateTensorFlowCode(layers);
    }
}

function switchFramework(framework) {
    currentFramework = framework;
    document.querySelectorAll('.code-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
    updateVisualization();
}

// הוספה לקובץ script.js

function updateCalculations(layers) {
    let weightsCalc = [];
    let biasesCalc = [];
    let totalWeights = 0;
    let totalBiases = 0;

    for (let i = 0; i < layers.length - 1; i++) {
        const weights = layers[i] * layers[i + 1];
        const biases = layers[i + 1];
        
        weightsCalc.push(`שכבה ${i} → ${i + 1}: ${layers[i]} × ${layers[i + 1]} = ${weights}`);
        biasesCalc.push(`שכבה ${i + 1}: ${biases} נוירונים = ${biases} הטיות`);
        
        totalWeights += weights;
        totalBiases += biases;
    }

    document.getElementById('weights-calculation').innerHTML = 
        weightsCalc.join('<br>') + 
        `<br>סה"כ: ${totalWeights}`;
    
    document.getElementById('biases-calculation').innerHTML = 
        biasesCalc.join('<br>') + 
        `<br>סה"כ: ${totalBiases}`;
    
    document.getElementById('params-calculation').innerHTML = 
        `משקולות: ${totalWeights}<br>` +
        `הטיות: ${totalBiases}<br>` +
        `סה"כ: ${totalWeights + totalBiases}`;
}

function initializeTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            const contentId = `${tab.dataset.tab}-content`;
            document.getElementById(contentId).classList.add('active');
        });
    });

    // Initialize terms content
    const termsContainer = document.getElementById('terms-content');
    termsContainer.innerHTML = ''; // נקה תוכן קיים
    Object.entries(DICTIONARY_CONTENT).forEach(([term, data]) => {
        termsContainer.innerHTML += `
            <div class="term-card">
                <div class="term-title">
                    <i class="fas ${data.icon}"></i>
                    ${term}
                </div>
                <div class="term-content">${data.content}</div>
            </div>
        `;
    });

    // Initialize metrics content
    const metricsContainer = document.getElementById('metrics-content');
    metricsContainer.innerHTML = ''; // נקה תוכן קיים
    Object.entries(METRICS_CONTENT).forEach(([metric, data]) => {
        metricsContainer.innerHTML += `
            <div class="metric-card">
                <div class="metric-title">
                    <i class="fas ${data.icon}"></i>
                    ${metric}
                </div>
                <div class="metric-content">
                    <div>נוסחה: ${data.formula}</div>
                    <div>${data.content}</div>
                    <div class="metric-example">
                        <div class="confusion-matrix">
                            ${renderConfusionMatrix(data.example.data)}
                        </div>
                        <div>חישוב: ${data.example.calculation}</div>
                    </div>
                </div>
            </div>
        `;
    });
}

// עדכון פונקציית initialize
function initialize() {
    if (initialized) return;
    
    initializeTabs();
    initializeCharts();
    updateVisualization();
    initializeDemoCanvas();  // הוספנו את זה
    
    // Set up event listeners
    document.getElementById('inputLayer').addEventListener('change', updateVisualization);
    document.getElementById('outputLayer').addEventListener('change', updateVisualization);
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('hidden-layer')) {
            updateVisualization();
        }
    });
    
    // הוספת האזנה לכפתורי ההדגמה
    document.querySelectorAll('.demo-button').forEach(button => {
        button.addEventListener('click', function() {
            const demoType = this.dataset.demo;
            demonstrateNetworkBehavior(demoType);
        });
    });
    
    initialized = true;
}

function renderConfusionMatrix(data) {
    return `
        <div class="matrix-cell matrix-header">True Positive</div>
        <div class="matrix-cell matrix-header">False Positive</div>
        <div class="matrix-cell">${data[0][0]}</div>
        <div class="matrix-cell">${data[0][1]}</div>
        <div class="matrix-cell">${data[1][0]}</div>
        <div class="matrix-cell">${data[1][1]}</div>
    `;
}

// עדכון פונקציית updateVisualization
function updateVisualization() {
    const layers = getLayers();
    drawNetwork(layers);
    updateStats();
    updateCalculations(layers);
    updateCode(layers);
}

// Initialize everything
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    updateVisualization();
});

// הוספה לקובץ script.js - הדמיית אימון

// משתנים גלובליים לאימון
let isTraining = false;
let trainingInterval;
let epoch = 0;
let lossHistory = [];
let dataPoints = [];
let currentPrediction = [];
let lossChart;
let dataChart;

// יצירת דאטה סינטטי
function generateData() {
    dataPoints = [];
    for (let i = 0; i < 50; i++) {
        const x = Math.random() * 4 - 2; // נקודות בין -2 ל 2
        const y = 0.5 * Math.sin(2 * x) + 0.1 * (Math.random() - 0.5);
        dataPoints.push({x, y});
    }
    currentPrediction = new Array(50).fill(0);
}

// יצירת גרפים
function initializeCharts() {
    // גרף Loss
    const lossCtx = document.getElementById('lossCanvas').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#6B46C1',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            }
        }
    });

    // גרף Data Fitting
    const dataCtx = document.getElementById('dataCanvas').getContext('2d');
    dataChart = new Chart(dataCtx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Training Data',
                    data: dataPoints.map(p => ({x: p.x, y: p.y})),
                    backgroundColor: '#9F7AEA',
                    pointRadius: 5
                },
                {
                    label: 'Model Prediction',
                    data: [],
                    borderColor: '#6B46C1',
                    borderWidth: 2,
                    pointRadius: 0,
                    type: 'line'
                }
            ]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'X'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Y'
                    }
                }
            }
        }
    });
}

// פונקציות אימון
function startTraining() {
    if (isTraining) return;
    
    isTraining = true;
    epoch = 0;
    lossHistory = [];
    generateData();
    
    document.getElementById('trainButton').disabled = true;
    document.getElementById('resetButton').disabled = false;
    
    // אתחול הגרפים
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    
    trainingInterval = setInterval(trainStep, 50);
}

function trainStep() {
    if (epoch >= 100) {
        stopTraining();
        return;
    }

    // הדמיית אימון
    epoch++;
    
    // חישוב Loss סינטטי
    const currentLoss = 1 / (1 + epoch * 0.1) + 0.1 * Math.random();
    lossHistory.push(currentLoss);
    
    // עדכון גרף Loss
    lossChart.data.labels.push(epoch);
    lossChart.data.datasets[0].data.push(currentLoss);
    lossChart.update();

    // עדכון תחזית המודל
    updatePrediction();
    
    // אנימציה של הרשת
    animateNetwork();
}

function updatePrediction() {
    // יצירת קו חלק של תחזיות
    const predictionPoints = [];
    for (let x = -2; x <= 2; x += 0.1) {
        const progress = Math.min(epoch / 100, 1);
        const trueY = 0.5 * Math.sin(2 * x);
        const noise = 0.5 * (1 - progress) * (Math.random() - 0.5);
        predictionPoints.push({
            x: x,
            y: trueY + noise
        });
    }
    
    dataChart.data.datasets[1].data = predictionPoints;
    dataChart.update();
}

function animateNetwork() {
    // הדגשת פעילות הנוירונים
    const layers = getLayers();
    const ctx = canvas.getContext('2d');
    
    // ציור הרשת עם אפקט זוהר
    drawNetwork(layers, {
        neuronGlow: true,
        weightAnimation: epoch
    });
}

function drawNetwork(layers, options = {}) {
    // שימוש בקוד הקיים של drawNetwork עם תוספות:
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    const padding = 50;
    const layerWidth = (width - 2 * padding) / (layers.length - 1);
    const neuronRadius = 12;
    
    ctx.clearRect(0, 0, width, height);
    
    // ציור קשרים
    for (let i = 0; i < layers.length - 1; i++) {
        const neurons = layers[i];
        const nextNeurons = layers[i + 1];
        const layerX = padding + i * layerWidth;
        const nextX = padding + (i + 1) * layerWidth;
        
        for (let j = 0; j < neurons; j++) {
            const neuronY = height/2 - (neurons * 30)/2 + j * 30;
            
            for (let k = 0; k < nextNeurons; k++) {
                const nextY = height/2 - (nextNeurons * 30)/2 + k * 30;
                
                // אנימציית משקולות
                if (options.weightAnimation) {
                    const phase = (options.weightAnimation * 0.1 + i * 0.5 + j * 0.2 + k * 0.3) % 1;
                    const opacity = 0.3 + 0.7 * phase;
                    ctx.strokeStyle = `rgba(159, 122, 234, ${opacity})`;
                    ctx.lineWidth = 1 + phase;
                } else {
                    ctx.strokeStyle = '#9F7AEA';
                    ctx.lineWidth = 0.5;
                }
                
                ctx.beginPath();
                ctx.moveTo(layerX + neuronRadius, neuronY);
                ctx.lineTo(nextX - neuronRadius, nextY);
                ctx.stroke();
            }
        }
    }
    
    // ציור נוירונים
    for (let i = 0; i < layers.length; i++) {
        const neurons = layers[i];
        const layerX = padding + i * layerWidth;
        
        for (let j = 0; j < neurons; j++) {
            const neuronY = height/2 - (neurons * 30)/2 + j * 30;
            
            ctx.beginPath();
            ctx.arc(layerX, neuronY, neuronRadius, 0, 2 * Math.PI);
            
            // אפקט זוהר בזמן אימון
            if (options.neuronGlow) {
                const gradient = ctx.createRadialGradient(
                    layerX, neuronY, 0,
                    layerX, neuronY, neuronRadius * 1.5
                );
                const phase = (epoch * 0.1 + i * 0.3 + j * 0.2) % 1;
                gradient.addColorStop(0, '#9F7AEA');
                gradient.addColorStop(0.6, '#6B46C1');
                gradient.addColorStop(1, `rgba(107, 70, 193, 0)`);
                ctx.fillStyle = gradient;
                
                ctx.shadowColor = '#6B46C1';
                ctx.shadowBlur = 15 * phase;
            } else {
                const gradient = ctx.createRadialGradient(
                    layerX, neuronY, 0,
                    layerX, neuronY, neuronRadius
                );
                gradient.addColorStop(0, '#9F7AEA');
                gradient.addColorStop(1, '#6B46C1');
                ctx.fillStyle = gradient;
            }
            
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }
}

function stopTraining() {
    isTraining = false;
    clearInterval(trainingInterval);
    document.getElementById('trainButton').disabled = false;
}

function resetTraining() {
    stopTraining();
    epoch = 0;
    lossHistory = [];
    currentPrediction = [];
    
    // איפוס גרפים
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update();
    
    dataChart.data.datasets[1].data = [];
    dataChart.update();
    
    // ציור מחדש של הרשת
    updateVisualization();
    
    document.getElementById('resetButton').disabled = true;
}

// נוסיף משתנים גלובליים חדשים
let demonstrationMode = null;
let demonstrationData = null;
let demonstrationInterval;
let demonstrationStep = 0;

function createDemonstrationData(type) {
    switch(type) {
        case 'classification':
            // יצירת דאטה של ספרות בכתב יד (מפושט)
            return {
                digits: [
                    { pixels: createDigitPixels('0'), label: 0 },
                    { pixels: createDigitPixels('1'), label: 1 },
                    { pixels: createDigitPixels('2'), label: 2 }
                ],
                currentDigit: 0,
                predictions: []
            };
        case 'regression':
            // יצירת נקודות לרגרסיה
            return {
                points: Array.from({length: 50}, (_, i) => ({
                    x: (i - 25) / 10,
                    y: Math.sin((i - 25) / 5) + Math.random() * 0.5
                })),
                currentFit: []
            };
        case 'clustering':
            // יצירת נקודות לקלסטרינג
            return {
                points: createClusterPoints(),
                clusters: [],
                centroids: []
            };
        case 'overfitting':
            // יצירת דאטה להדגמת Overfitting
            return {
                trainingData: createOverfittingData('train'),
                testData: createOverfittingData('test'),
                trainLoss: [],
                testLoss: [],
                complexity: 1
            };
    }
}

function demonstrateNetworkBehavior(type) {
    // ניקוי מצב קודם
    if (demonstrationInterval) {
        clearInterval(demonstrationInterval);
    }
    
    demonstrationMode = type;
    demonstrationStep = 0;
    demonstrationData = createDemonstrationData(type);
    
    // התאמת ממשק המשתמש
    updateDemonstrationUI(type);
    
    // התחלת הדגמה
    demonstrationInterval = setInterval(() => {
        switch(type) {
            case 'classification':
                simulateDigitClassification();
                break;
            case 'regression':
                simulateRegression();
                break;
            case 'clustering':
                simulateClustering();
                break;
            case 'overfitting':
                simulateOverfitting();
                break;
        }
        demonstrationStep++;
    }, 100);
}

function simulateDigitClassification() {
    const { digits, currentDigit } = demonstrationData;
    
    // הדמיית תהליך הסיווג
    if (demonstrationStep % 20 === 0) {
        // החלפת הספרה הנוכחית
        demonstrationData.currentDigit = (currentDigit + 1) % digits.length;
    }
    
    // חישוב "הסתברויות" מדומות לכל ספרה
    const predictions = Array.from({length: 10}, (_, i) => {
        const isCorrect = i === digits[currentDigit].label;
        const progress = Math.min((demonstrationStep % 20) / 10, 1);
        return isCorrect ? 
            0.1 + 0.8 * progress : 
            0.1 * (1 - progress) + Math.random() * 0.1;
    });
    
    demonstrationData.predictions = predictions;
    
    // עדכון הויזואליזציה
    drawClassificationDemo();
}

function simulateRegression() {
    const { points } = demonstrationData;
    
    // יצירת קו התאמה שמשתפר עם הזמן
    const progress = Math.min(demonstrationStep / 50, 1);
    const fit = points.map(point => ({
        x: point.x,
        y: predictRegressionValue(point.x, progress)
    }));
    
    demonstrationData.currentFit = fit;
    
    // עדכון הויזואליזציה
    drawRegressionDemo();
}

function simulateClustering() {
    const canvas = document.getElementById('demoCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // יצירת נקודות רנדומליות בשלושה אשכולות
    if (!demonstrationData.clusters) {
        demonstrationData.clusters = [
            { center: { x: canvas.width * 0.2, y: canvas.height * 0.3 }, points: [] },
            { center: { x: canvas.width * 0.7, y: canvas.height * 0.7 }, points: [] },
            { center: { x: canvas.width * 0.8, y: canvas.height * 0.2 }, points: [] }
        ];

        // יצירת נקודות סביב כל מרכז
        demonstrationData.clusters.forEach(cluster => {
            for (let i = 0; i < 30; i++) {
                cluster.points.push({
                    x: cluster.center.x + (Math.random() - 0.5) * 100,
                    y: cluster.center.y + (Math.random() - 0.5) * 100
                });
            }
        });
    }

    // הזזת מרכזי האשכולות בהדרגה
    demonstrationData.clusters.forEach((cluster, i) => {
        // עדכון מיקום המרכז
        const newCenter = {
            x: 0,
            y: 0
        };
        cluster.points.forEach(point => {
            newCenter.x += point.x;
            newCenter.y += point.y;
        });
        newCenter.x /= cluster.points.length;
        newCenter.y /= cluster.points.length;
        
        // הזזה הדרגתית של המרכז
        cluster.center.x += (newCenter.x - cluster.center.x) * 0.1;
        cluster.center.y += (newCenter.y - cluster.center.y) * 0.1;

        // ציור הנקודות
        cluster.points.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = ['#9F7AEA', '#4FD1C5', '#F6AD55'][i];
            ctx.fill();
        });

        // ציור מרכז האשכול
        ctx.beginPath();
        ctx.arc(cluster.center.x, cluster.center.y, 10, 0, Math.PI * 2);
        ctx.strokeStyle = ['#6B46C1', '#319795', '#DD6B20'][i];
        ctx.lineWidth = 3;
        ctx.stroke();
    });
}

function simulateOverfitting() {
    const canvas = document.getElementById('demoCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // יצירת נקודות אימון אם לא קיימות
    if (!demonstrationData.trainPoints) {
        demonstrationData.trainPoints = Array.from({length: 20}, (_, i) => ({
            x: i * (canvas.width / 20),
            y: (canvas.height / 2) + Math.sin(i / 3) * 50 + (Math.random() - 0.5) * 30
        }));
    }

    // ציור נקודות האימון
    demonstrationData.trainPoints.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#9F7AEA';
        ctx.fill();
    });

    // חישוב מורכבות פעם אחת
    const complexity = Math.min(demonstrationStep / 50, 1) * 10;

    // ציור קו ההתאמה - מורכבות עולה עם הזמן
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x++) {
        let y = canvas.height / 2;
        for (let i = 1; i <= complexity; i++) {
            y += Math.sin(x * i / 50) * (20 / i);
        }
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.strokeStyle = '#6B46C1';
    ctx.lineWidth = 2;
    ctx.stroke();

    // עדכון גרף ה-Loss
    updateLossGraph(complexity);  // משתמשים בפונקציה המקורית
}


// פונקציות עזר

function drawRegressionPoints(ctx, points) {
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(
            ctx.canvas.width/2 + point.x * 50,  // נירמול לגודל הקנבס
            ctx.canvas.height/2 - point.y * 50,  // הפוך את ציר ה-y
            4,
            0,
            2 * Math.PI
        );
        ctx.fillStyle = '#9F7AEA';
        ctx.fill();
    });
}

function drawRegressionLine(ctx, points, progress) {
    ctx.beginPath();
    ctx.strokeStyle = '#6B46C1';
    ctx.lineWidth = 2;
    
    // יצירת קו רציף של חיזויים
    const xPoints = Array.from({length: 100}, (_, i) => -5 + i * 0.1);
    
    xPoints.forEach((x, i) => {
        const predictedY = predictRegressionValue(x, progress);
        const canvasX = ctx.canvas.width/2 + x * 50;
        const canvasY = ctx.canvas.height/2 - predictedY * 50;
        
        if (i === 0) {
            ctx.moveTo(canvasX, canvasY);
        } else {
            ctx.lineTo(canvasX, canvasY);
        }
    });
    
    ctx.stroke();
}

function calculateRegressionLoss(progress) {
    const trueFunction = x => Math.sin(x);
    const currentPrediction = x => trueFunction(x) + (1 - progress) * (Math.random() - 0.5);
    
    // חישוב MSE על מספר נקודות
    let totalError = 0;
    const numPoints = 20;
    
    for (let i = 0; i < numPoints; i++) {
        const x = -5 + i * 0.5;
        const trueY = trueFunction(x);
        const predY = currentPrediction(x);
        totalError += Math.pow(trueY - predY, 2);
    }
    
    return totalError / numPoints;
}

function createDigitPixels(digit) {
    const digit_patterns = {
        0: [[1,1,1], [1,0,1], [1,0,1], [1,0,1], [1,1,1]],
        1: [[0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]],
        2: [[1,1,1], [0,0,1], [1,1,1], [1,0,0], [1,1,1]],
        3: [[1,1,1], [0,0,1], [1,1,1], [0,0,1], [1,1,1]],
        4: [[1,0,1], [1,0,1], [1,1,1], [0,0,1], [0,0,1]],
        5: [[1,1,1], [1,0,0], [1,1,1], [0,0,1], [1,1,1]],
        6: [[1,1,1], [1,0,0], [1,1,1], [1,0,1], [1,1,1]],
        7: [[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1]],
        8: [[1,1,1], [1,0,1], [1,1,1], [1,0,1], [1,1,1]],
        9: [[1,1,1], [1,0,1], [1,1,1], [0,0,1], [1,1,1]]
    };
    return digit_patterns[digit];
}

function addTypeSpecificLabels(ctx, type, padding, height) {
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    
    switch(type) {
        case 'classification':
            ctx.fillText('Classification Accuracy', padding - 5, padding - 10);
            break;
        case 'regression':
            ctx.fillText('Mean Squared Error', padding - 5, padding - 10);
            break;
        case 'clustering':
            ctx.fillText('Inertia', padding - 5, padding - 10);
            break;
    }
}

function drawAxes(ctx, padding, width, height) {
    ctx.beginPath();
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    
    // Y axis
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height + padding);
    
    // X axis
    ctx.moveTo(padding, height + padding);
    ctx.lineTo(width + padding, height + padding);
    
    ctx.stroke();
}

function drawDigit(ctx, pixels, x, y, size = 20) {
    pixels.forEach((row, i) => {
        row.forEach((val, j) => {
            ctx.fillStyle = val ? '#6B46C1' : '#EDF2F7';
            ctx.fillRect(x + j * size, y + i * size, size, size);
        });
    });
}

function simulateClassification() {
    // יצירת דוגמה של ספרה אם לא קיימת
    if (!demonstrationData.currentDigit) {
        demonstrationData.currentDigit = createDigitExample();
        demonstrationData.predictions = new Array(10).fill(0);
    }

    const demoCanvas = document.getElementById('demoCanvas');
    const ctx = demoCanvas.getContext('2d');
    
    // ציור הספרה
    drawDigit(ctx, demonstrationData.currentDigit);
    
    // עדכון הסתברויות
    const targetDigit = demonstrationData.currentDigit.label;
    const progress = Math.min(demonstrationStep / 20, 1);
    
    demonstrationData.predictions = demonstrationData.predictions.map((_, i) => {
        if (i === targetDigit) {
            return 0.1 + 0.8 * progress;
        } else {
            return 0.1 * (1 - progress) + 0.05 * Math.random();
        }
    });

    // ציור ההסתברויות
    drawPredictionBars(ctx, demonstrationData.predictions);

    // עדכון גרף Loss
    const currentLoss = 1 - demonstrationData.predictions[targetDigit];
    if (!demonstrationData.lossHistory) {
        demonstrationData.lossHistory = [];
    }
    demonstrationData.lossHistory.push(currentLoss);
    
    const lossCanvas = document.getElementById('lossCanvas');
    drawLossHistory(lossCanvas.getContext('2d'), demonstrationData.lossHistory);
}

function createDigitExample() {
    // יצירת מטריצת פיקסלים של ספרה
    return {
        pixels: [
            [0,1,1,0],
            [1,0,0,1],
            [1,0,0,1],
            [0,1,1,0]
        ],
        label: 2  // לדוגמה: ספרה 2
    };
}

function drawDigit(ctx, digit) {
    const pixelSize = 20;
    ctx.clearRect(0, 0, 200, 200);
    
    digit.pixels.forEach((row, i) => {
        row.forEach((value, j) => {
            ctx.fillStyle = value ? '#6B46C1' : '#EDF2F7';
            ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
        });
    });
}

function drawPredictionBars(ctx, predictions) {
    const barHeight = 20;
    const maxWidth = 150;
    
    ctx.save();
    ctx.translate(220, 0);  // הזזה ימינה לציור ההסתברויות
    
    predictions.forEach((prob, i) => {
        const y = i * (barHeight + 5);
        
        // ציור הרקע
        ctx.fillStyle = '#EDF2F7';
        ctx.fillRect(0, y, maxWidth, barHeight);
        
        // ציור ההסתברות
        ctx.fillStyle = '#6B46C1';
        ctx.fillRect(0, y, prob * maxWidth, barHeight);
        
        // הוספת תווית
        ctx.fillStyle = '#1A202C';
        ctx.fillText(`${i}: ${(prob * 100).toFixed(1)}%`, maxWidth + 10, y + barHeight/2);
    });
    
    ctx.restore();
}

function simulateRegression() {
    if (!demonstrationData.regressionPoints) {
        demonstrationData.regressionPoints = generateRegressionData();
    }

    const demoCanvas = document.getElementById('demoCanvas');
    const ctx = demoCanvas.getContext('2d');
    
    // ציור נקודות האימון
    drawRegressionPoints(ctx, demonstrationData.regressionPoints);
    
    // ציור קו החיזוי
    const progress = Math.min(demonstrationStep / 50, 1);
    drawRegressionLine(ctx, demonstrationData.regressionPoints, progress);

    // עדכון Loss
    const currentLoss = calculateRegressionLoss(progress);
    if (!demonstrationData.lossHistory) {
        demonstrationData.lossHistory = [];
    }
    demonstrationData.lossHistory.push(currentLoss);
    
    const lossCanvas = document.getElementById('lossCanvas');
    drawLossHistory(lossCanvas.getContext('2d'), demonstrationData.lossHistory);
}

function generateRegressionData() {
    return Array.from({length: 20}, (_, i) => ({
        x: (i - 10) / 5,
        y: Math.sin((i - 10) / 3) + 0.2 * (Math.random() - 0.5)
    }));
}

function updateLossGraphForType(type) {
    const lossCanvas = document.getElementById('lossCanvas');
    const ctx = lossCanvas.getContext('2d');
    
    if (!demonstrationData.lossHistory) {
        demonstrationData.lossHistory = [];
    }
    
    let currentLoss;
    switch(type) {
        case 'classification':
            // Loss יורד מהר בהתחלה ואז מתייצב
            currentLoss = 1 / (1 + demonstrationStep * 0.1) + 0.1 * Math.random();
            break;
            
        case 'regression':
            // Loss יורד בהדרגה עם תנודות קטנות
            currentLoss = 0.5 / (1 + demonstrationStep * 0.05) + 
                         0.05 * Math.sin(demonstrationStep * 0.1) +
                         0.05 * Math.random();
            break;
            
        case 'clustering':
            // Loss יורד בקפיצות כשהאשכולות מתייצבים
            currentLoss = 0.8 * Math.exp(-demonstrationStep * 0.05) +
                         0.2 * Math.random();
            break;
            
        case 'overfitting':
            // קריאה לפונקציה המקורית עבור overfitting
            const complexity = Math.min(demonstrationStep / 50, 1) * 10;
            updateLossGraph(complexity);
            return;
    }
    
    demonstrationData.lossHistory.push(currentLoss);
    
    // ציור הגרף הבסיסי
    drawBasicLossGraph(demonstrationData.lossHistory);
}

function drawBasicLossGraph(history) {
    const lossCanvas = document.getElementById('lossCanvas');
    const ctx = lossCanvas.getContext('2d');
    
    ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    
    const padding = 40;
    const width = lossCanvas.width - 2 * padding;
    const height = lossCanvas.height - 2 * padding;
    
    // ציור צירים
    ctx.beginPath();
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    
    // ציר Y
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, lossCanvas.height - padding);
    
    // ציר X
    ctx.moveTo(padding, lossCanvas.height - padding);
    ctx.lineTo(lossCanvas.width - padding, lossCanvas.height - padding);
    ctx.stroke();
    
    // תוויות
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    
    ctx.fillText('Loss', padding - 5, padding - 10);
    ctx.fillText('1.0', padding - 5, padding);
    ctx.fillText('0.5', padding - 5, padding + height/2);
    ctx.fillText('0.0', padding - 5, padding + height);
    
    ctx.textAlign = 'center';
    ctx.fillText('Epochs', lossCanvas.width/2, lossCanvas.height - 5);
    
    // ציור הגרף
    ctx.beginPath();
    ctx.strokeStyle = '#6B46C1';
    ctx.lineWidth = 2;
    
    history.forEach((loss, i) => {
        const x = padding + (i / (history.length - 1)) * width;
        const y = padding + (1 - loss) * height;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
}




function simulateClustering() {
    // אתחול הדאטה אם לא קיים
    if (!demonstrationData.clusters) {
        demonstrationData.clusters = initializeClusters();
    }

    const demoCanvas = document.getElementById('demoCanvas');
    const ctx = demoCanvas.getContext('2d');
    ctx.clearRect(0, 0, demoCanvas.width, demoCanvas.height);

    // עדכון מיקומי המרכזים
    updateClusterCenters();

    // שיוך נקודות מחדש לאשכולות
    assignPointsToClusters();

    // ציור האשכולות
    drawClusters(ctx);

    // עדכון גרף ה-Loss
    updateClusteringLoss();
}


function updateLossGraph(complexity) {
    const lossCanvas = document.getElementById('lossCanvas');
    const ctx = lossCanvas.getContext('2d');
    
    // נקה את הקנבס
    ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    
    // חישוב ערכי Loss
    const trainLoss = 1 / (1 + complexity);
    const testLoss = complexity > 5 ? 
        Math.pow(complexity - 5, 2) / 20 : 
        1 / (1 + complexity);
    
    // הוספת ערכים להיסטוריה
    if (!demonstrationData.lossHistory) {
        demonstrationData.lossHistory = {
            train: [],
            test: []
        };
    }
    
    demonstrationData.lossHistory.train.push(trainLoss);
    demonstrationData.lossHistory.test.push(testLoss);
    
    // הגדרת סגנון ציור
    const padding = 40;
    const width = lossCanvas.width - 2 * padding;
    const height = lossCanvas.height - 2 * padding;
    
    // ציור צירים
    ctx.beginPath();
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    
    // ציר Y
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, lossCanvas.height - padding);
    
    // ציר X
    ctx.moveTo(padding, lossCanvas.height - padding);
    ctx.lineTo(lossCanvas.width - padding, lossCanvas.height - padding);
    ctx.stroke();
    
    // הוספת תוויות
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    
    // תוויות ציר Y
    ctx.fillText('Loss', padding - 5, padding - 10);
    ctx.fillText('1.0', padding - 5, padding);
    ctx.fillText('0.5', padding - 5, padding + height/2);
    ctx.fillText('0.0', padding - 5, padding + height);
    
    // תוויות ציר X
    ctx.textAlign = 'center';
    ctx.fillText('Model Complexity', lossCanvas.width/2, lossCanvas.height - 5);
    
    // ציור הגרפים
    const history = demonstrationData.lossHistory;
    const pointsToPixels = (points, color) => {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        
        points.forEach((loss, i) => {
            const x = padding + (i / (points.length - 1)) * width;
            const y = padding + (1 - loss) * height;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
    };
    
    // ציור גרף אימון
    pointsToPixels(history.train, '#6B46C1');
    
    // ציור גרף בדיקה
    pointsToPixels(history.test, '#DD6B20');
    
    // הוספת מקרא
    const legendY = padding + 20;
    ctx.textAlign = 'left';
    
    // מקרא לגרף אימון
    ctx.fillStyle = '#6B46C1';
    ctx.fillRect(padding + width - 100, legendY, 20, 2);
    ctx.fillStyle = '#666';
    ctx.fillText('Training Loss', padding + width - 70, legendY + 5);
    
    // מקרא לגרף בדיקה
    ctx.fillStyle = '#DD6B20';
    ctx.fillRect(padding + width - 100, legendY + 20, 20, 2);
    ctx.fillStyle = '#666';
    ctx.fillText('Test Loss', padding + width - 70, legendY + 25);
    
    // הוספת הערה על Overfitting
    if (complexity > 5) {
        const overfit = Math.min((complexity - 5) / 5, 1);
        ctx.globalAlpha = overfit;
        ctx.fillStyle = '#E53E3E';
        ctx.textAlign = 'center';
        ctx.fillText('Overfitting!', lossCanvas.width/2, padding + 50);
        ctx.globalAlpha = 1;
    }
    
    // הוספת המלצה למורכבות אופטימלית
    if (complexity >= 5 && complexity <= 6) {
        ctx.fillStyle = '#38A169';
        ctx.textAlign = 'center';
        ctx.fillText('Sweet Spot', lossCanvas.width/2, padding + 70);
    }
}

function predictRegressionValue(x, progress) {
    // חישוב ערך חיזוי שמשתפר עם הזמן
    const trueValue = Math.sin(x);
    const noise = (1 - progress) * (Math.random() - 0.5);
    return trueValue + noise;
}

function createClusterPoints() {
    // יצירת נקודות בשלושה אשכולות
    const clusters = [
        {center: {x: -2, y: -2}, radius: 1},
        {center: {x: 2, y: 2}, radius: 1.5},
        {center: {x: -1, y: 3}, radius: 1.2}
    ];
    
    return clusters.flatMap(cluster => 
        Array.from({length: 20}, () => ({
            x: cluster.center.x + (Math.random() - 0.5) * cluster.radius,
            y: cluster.center.y + (Math.random() - 0.5) * cluster.radius
        }))
    );
}

function createOverfittingData(type) {
    // יצירת דאטה להדגמת Overfitting
    const basePoints = Array.from({length: 20}, (_, i) => ({
        x: (i - 10) / 5,
        y: Math.sin(i / 3) + (type === 'train' ? 0.2 : -0.2)
    }));
    
    return basePoints.map(point => ({
        ...point,
        y: point.y + (Math.random() - 0.5) * 0.3
    }));
}

// פונקציות ויזואליזציה

function drawClassificationDemo() {
    const ctx = document.getElementById('dataCanvas').getContext('2d');
    const { digits, currentDigit, predictions } = demonstrationData;
    
    // ציור הספרה
    const pixelSize = 20;
    const pixels = digits[currentDigit].pixels;
    
    ctx.clearRect(0, 0, 400, 400);
    
    // ציור מטריצת הפיקסלים
    pixels.forEach((row, i) => {
        row.forEach((value, j) => {
            ctx.fillStyle = value ? '#6B46C1' : '#EDF2F7';
            ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
        });
    });
    
    // ציור הסתברויות
    predictions.forEach((prob, i) => {
        ctx.fillStyle = '#6B46C1';
        ctx.fillRect(200, i * 30, prob * 150, 20);
        ctx.fillStyle = '#1A202C';
        ctx.fillText(`${i}: ${(prob * 100).toFixed(1)}%`, 360, i * 30 + 15);
    });
}

function drawLossHistory(history, type) {
    const lossCanvas = document.getElementById('lossCanvas');
    const ctx = lossCanvas.getContext('2d');
    
    ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    
    const padding = 40;
    const width = lossCanvas.width - 2 * padding;
    const height = lossCanvas.height - 2 * padding;
    
    // ציור צירים וכותרות בסיסיות
    drawAxes(ctx, padding, width, height);
    
    // ציור הגרף
    ctx.beginPath();
    ctx.strokeStyle = '#6B46C1';
    ctx.lineWidth = 2;
    
    history.forEach((loss, i) => {
        const x = padding + (i / (history.length - 1)) * width;
        const y = padding + (1 - loss) * height;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    // הוספת תוויות ספציפיות לכל סוג
    addTypeSpecificLabels(ctx, type, padding, height);
}

function drawRegressionDemo() {
    const ctx = document.getElementById('dataCanvas').getContext('2d');
    const { points, currentFit } = demonstrationData;
    
    ctx.clearRect(0, 0, 400, 400);
    
    // ציור נקודות הדאטה
    points.forEach(point => {
        ctx.fillStyle = '#9F7AEA';
        ctx.beginPath();
        ctx.arc(
            200 + point.x * 50,
            200 - point.y * 50,
            4,
            0,
            2 * Math.PI
        );
        ctx.fill();
    });
    
    // ציור קו ההתאמה
    if (currentFit.length > 0) {
        ctx.strokeStyle = '#6B46C1';
        ctx.beginPath();
        ctx.moveTo(
            200 + currentFit[0].x * 50,
            200 - currentFit[0].y * 50
        );
        currentFit.forEach(point => {
            ctx.lineTo(
                200 + point.x * 50,
                200 - point.y * 50
            );
        });
        ctx.stroke();
    }
}

function drawClusteringDemo() {
    const ctx = document.getElementById('dataCanvas').getContext('2d');
    const { points, clusters, centroids } = demonstrationData;
    
    ctx.clearRect(0, 0, 400, 400);
    
    // ציור נקודות
    points.forEach((point, i) => {
        const cluster = clusters[i] || 0;
        ctx.fillStyle = ['#9F7AEA', '#4FD1C5', '#F6AD55'][cluster];
        ctx.beginPath();
        ctx.arc(
            200 + point.x * 50,
            200 - point.y * 50,
            4,
            0,
            2 * Math.PI
        );
        ctx.fill();
    });
    
    // ציור צנטרואידים
    centroids.forEach((centroid, i) => {
        ctx.strokeStyle = ['#6B46C1', '#319795', '#DD6B20'][i];
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(
            200 + centroid.x * 50,
            200 - centroid.y * 50,
            8,
            0,
            2 * Math.PI
        );
        ctx.stroke();
    });
}

function drawOverfittingDemo() {
    const ctx = document.getElementById('dataCanvas').getContext('2d');
    const { trainingData, testData, trainLoss, testLoss } = demonstrationData;
    
    ctx.clearRect(0, 0, 400, 400);
    
    // ציור נקודות אימון
    trainingData.forEach(point => {
        ctx.fillStyle = '#9F7AEA';
        ctx.beginPath();
        ctx.arc(
            200 + point.x * 50,
            200 - point.y * 50,
            4,
            0,
            2 * Math.PI
        );
        ctx.fill();
    });
    
    // ציור נקודות טסט
    testData.forEach(point => {
        ctx.fillStyle = '#F6AD55';
        ctx.beginPath();
        ctx.arc(
            200 + point.x * 50,
            200 - point.y * 50,
            4,
            0,
            2 * Math.PI
        );
        ctx.fill();
    });
    
    // ציור גרף Loss
    const lossCtx = document.getElementById('lossCanvas').getContext('2d');
    lossCtx.clearRect(0, 0, 400, 400);
    
    // ציור Loss אימון
    lossCtx.strokeStyle = '#6B46C1';
    lossCtx.beginPath();
    trainLoss.forEach((loss, i) => {
        if (i === 0) {
            lossCtx.moveTo(i * 2, 200 - loss * 100);
        } else {
            lossCtx.lineTo(i * 2, 200 - loss * 100);
        }
    });
    lossCtx.stroke();
    
    // ציור Loss טסט
    lossCtx.strokeStyle = '#DD6B20';
    lossCtx.beginPath();
    testLoss.forEach((loss, i) => {
        if (i === 0) {
            lossCtx.moveTo(i * 2, 200 - loss * 100);
        } else {
            lossCtx.lineTo(i * 2, 200 - loss * 100);
        }
    });
    lossCtx.stroke();
}

// הוספה לקובץ JavaScript

const DEMO_EXPLANATIONS = {
    'classification': {
        title: 'זיהוי ספרות בכתב יד',
        content: `הדגמה זו מראה כיצד רשת נוירונים מזהה ספרות בכתב יד:
        • כל פיקסל בתמונה הוא קלט לרשת
        • הרשת מחשבת הסתברות לכל ספרה (0-9)
        • הערך הגבוה ביותר הוא הניחוש של הרשת
        • שים לב איך ההסתברויות מתעדכנות בזמן אמת`
    },
    'regression': {
        title: 'חיזוי ערכים רציפים',
        content: `הדגמה של רגרסיה - חיזוי ערך רציף:
        • הנקודות הכחולות הן דוגמאות האימון
        • הקו הסגול מייצג את החיזוי של הרשת
        • ראה איך הרשת לומדת להתאים את עצמה לדפוס בדאטה
        • זוהי משימה נפוצה בחיזוי מחירים, טמפרטורות וכו'`
    },
    'clustering': {
        title: 'קיבוץ נקודות לקבוצות',
        content: `הדגמת אלגוריתם קלסטרינג:
        • כל נקודה משויכת לאחת משלוש קבוצות
        • העיגולים הגדולים הם מרכזי הקבוצות
        • האלגוריתם מעדכן את המרכזים בהתאם לממוצע
        • שימושי לזיהוי דפוסים וסגמנטציה של לקוחות`
    },
    'overfitting': {
        title: 'התמודדות עם Overfitting',
        content: `הדגמה של בעיית ה-Overfitting:
        • נקודות סגולות: סט אימון
        • נקודות כתומות: סט בדיקה
        • גרף תחתון: שגיאת אימון (סגול) מול בדיקה (כתום)
        • שים לב לנקודה בה המודל מתחיל "לזכור" במקום ללמוד`
    }
};

function initializeDemoCanvas() {
    const demoCanvas = document.getElementById('demoCanvas');
    if (!demoCanvas) {
        console.error('Demo canvas not found');
        return;
    }
    
    // התאמת גודל הקנבס לגודל המכיל
    const container = demoCanvas.parentElement;
    demoCanvas.width = container.offsetWidth;
    demoCanvas.height = 400; // או גובה אחר מתאים
    
    // ציור רקע ראשוני
    const ctx = demoCanvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, demoCanvas.width, demoCanvas.height);
}

function updateDemonstrationUI(type) {
    const demoTitle = document.getElementById('demoTitle');
    const demoExplanation = document.getElementById('demoExplanation');
    
    const demoInfo = DEMO_EXPLANATIONS[type];
    demoTitle.textContent = demoInfo.title;
    demoExplanation.innerHTML = demoInfo.content;
}

function updateDemoMetrics(metrics) {
    const metricsContainer = document.getElementById('demoMetrics');
    metricsContainer.innerHTML = Object.entries(metrics)
        .map(([label, value]) => `
            <div class="metric-item">
                <div class="metric-value">${value}</div>
                <div class="metric-label">${label}</div>
            </div>
        `).join('');
}

let initialized = false;

function initialize() {
    if (initialized) return;
    
    initializeTabs();
    initializeCharts();
    updateVisualization();
    
    // Set up event listeners
    document.getElementById('inputLayer').addEventListener('change', updateVisualization);
    document.getElementById('outputLayer').addEventListener('change', updateVisualization);
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('hidden-layer')) {
            updateVisualization();
        }
    });
    
    initialized = true;
}

// אתחול בטעינת הדף
document.addEventListener('DOMContentLoaded', initialize);

