document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('score-form');
    const apiUrlInput = document.getElementById('api-url');
    const imageUrlInput = document.getElementById('image-url');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const errorMessage = document.getElementById('error-message');
    const resultSection = document.getElementById('result-section');

    // Result Elements
    const resTotal = document.getElementById('res-total');
    const resProb = document.getElementById('res-prob');
    const resCaseId = document.getElementById('res-caseid');
    const resTime = document.getElementById('res-time');

    // Bar elements
    const bars = {
        'O': { fill: document.getElementById('bar-o'), val: document.getElementById('val-o') },
        'X': { fill: document.getElementById('bar-x'), val: document.getElementById('val-x') },
        'S': { fill: document.getElementById('bar-s'), val: document.getElementById('val-s') },
        'T': { fill: document.getElementById('bar-t'), val: document.getElementById('val-t') },
        'D': { fill: document.getElementById('bar-d'), val: document.getElementById('val-d') }
    };

    // Live Image Preview
    // Live Image Preview
    const updateImagePreview = (url) => {
        if (url && url.trim() !== '') {
            imagePreview.src = url;
            imagePreviewContainer.style.display = 'block';
        } else {
            imagePreviewContainer.style.display = 'none';
        }
    };

    imageUrlInput.addEventListener('input', (e) => {
        updateImagePreview(e.target.value);
    });

    imagePreview.addEventListener('error', () => {
        // If image fails to load, maybe hide or show error placeholder
        // imagePreviewContainer.style.display = 'none'; 
        // Keeping it might show broken image icon which is informative
    });

    // Trigger preview on load if default token exists
    // Trigger preview on load if default val exists
    updateImagePreview(imageUrlInput.value);

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Reset state
        showError(null);
        setLoading(true);
        resultSection.style.display = 'none';

        // Get values
        const baseUrl = apiUrlInput.value.replace(/\/$/, ''); // Remove trailing slash
        const payload = {
            CaseID: document.getElementById('case-id').value,
            months: parseInt(document.getElementById('months').value),
            model: document.getElementById('model').value,
            imagepath: imageUrlInput.value
        };

        try {
            console.log(`Sending request to ${baseUrl}/score`, payload);

            const response = await fetch(`${baseUrl}/score`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `API Error: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error(error);
            showError(error.message || '發生未知錯誤，請檢查 console 詳情。');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        if (isLoading) {
            btnText.style.display = 'none';
            loader.style.display = 'inline-block';
        } else {
            btnText.style.display = 'inline-block';
            loader.style.display = 'none';
        }
    }

    function showError(msg) {
        if (msg) {
            errorMessage.textContent = msg;
            errorMessage.style.display = 'block';
        } else {
            errorMessage.style.display = 'none';
        }
    }

    function displayResults(data) {
        // Update Header Stats
        resCaseId.textContent = data.CaseID;
        resProb.textContent = Number(data.Probability).toFixed(2);
        resTime.textContent = new Date(data.Timestamp).toLocaleString();

        // Total Score
        const scores = data.Score;
        resTotal.textContent = scores.Score.toFixed(2); // "Score" key in "Score" object

        // Update Bars
        const maxScore = 20; // Assuming max possible score roughly to calculate percentage, or just use raw if small.
        // Actually scores seem specific: T=8, D=16, S=4, X=2, O=1. Total around 31?
        // Let's normalize visually based on an assumed max or just proportional.
        // Let's assume max visible bar is 16 (D's max).

        const updateBar = (key) => {
            const val = scores[key] || 0;
            // Let's cap visual width at 100% for reasonable viewing
            // Max score per shape varies, let's just use 100% = 20 for scaling visual
            const percentage = Math.min((val / 16) * 100, 100);

            bars[key].val.textContent = val.toFixed(2);
            bars[key].fill.style.width = `${percentage}%`;
        };

        ['O', 'X', 'S', 'T', 'D'].forEach(updateBar);

        // Show section
        resultSection.style.display = 'block';

        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
});
