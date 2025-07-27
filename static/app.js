document.addEventListener('DOMContentLoaded', () => {
    const token = sessionStorage.getItem('accessToken');
    if (token) {
        showApp();
        fetchPdfs(); // Fetch PDFs on load if logged in
    }
});

// --- STATE MANAGEMENT ---
let accessToken = sessionStorage.getItem('accessToken') || null;
let currentPdfId = null;
let currentConversationId = null;

// --- DOM ELEMENTS ---
const authContainer = document.getElementById('auth-container');
const appContainer = document.getElementById('app-container');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const chatHistory = document.getElementById('chat-history');
const pdfList = document.getElementById('pdf-list');
const chatHeader = document.getElementById('chat-header');

// --- UI TOGGLING ---
function toggleAuthForms() {
    loginForm.style.display = loginForm.style.display === 'none' ? 'block' : 'none';
    registerForm.style.display = registerForm.style.display === 'none' ? 'block' : 'none';
}

function showApp() {
    authContainer.style.display = 'none';
    appContainer.style.display = 'block';
}

function showAuth() {
    sessionStorage.removeItem('accessToken');
    accessToken = null;
    authContainer.style.display = 'block';
    appContainer.style.display = 'none';
}

// --- API CALLS ---

async function register() {
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;
    if (!username || !password) {
        alert('Please enter both username and password.');
        return;
    }

    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        if (response.status === 201) {
            alert('Registration successful! Please login.');
            toggleAuthForms();
        } else {
            const data = await response.json();
            alert(`Registration failed: ${data.msg}`);
        }
    } catch (error) {
        alert('An error occurred during registration.');
        console.error('Registration Error:', error);
    }
}

async function login() {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    if (!username || !password) {
        alert('Please enter both username and password.');
        return;
    }

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        if (response.ok) {
            const data = await response.json();
            accessToken = data.access_token;
            sessionStorage.setItem('accessToken', accessToken);
            showApp();
            fetchPdfs(); // Fetch PDFs on login
        } else {
            alert('Login failed. Please check your credentials.');
        }
    } catch (error) {
        alert('An error occurred during login.');
        console.error('Login Error:', error);
    }
}

async function uploadPdf() {
    const pdfFileInput = document.getElementById('pdf-file-input');
    if (pdfFileInput.files.length === 0) {
        alert('Please select a PDF file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('pdf_file', pdfFileInput.files[0]);

    try {
        const response = await fetch('/upload_pdf', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${accessToken}` },
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            currentPdfId = data.pdf_id;
            currentConversationId = null; // Reset conversation for new PDF
            chatHistory.innerHTML = ''; // Clear chat history
            alert(`PDF processed successfully! You can now chat with it.`);
            fetchPdfs(); // Refresh the PDF list
            selectPdf(currentPdfId, pdfFileInput.files[0].name);
        } else {
            alert('PDF upload failed.');
        }
    } catch (error) {
        alert('An error occurred during PDF upload.');
        console.error('Upload Error:', error);
    }
}

async function askPdf() {
    const userQuery = document.getElementById('user-query').value;
    if (!userQuery) {
        alert('Please enter a question.');
        return;
    }
    if (!currentPdfId) {
        alert('Please upload a PDF first.');
        return;
    }

    appendMessage(userQuery, 'user');
    document.getElementById('user-query').value = '';

    try {
        const response = await fetch('/ask_pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${accessToken}`
            },
            body: JSON.stringify({
                query: userQuery,
                pdf_id: currentPdfId,
                conversation_id: currentConversationId
            })
        });

        if (response.ok) {
            const data = await response.json();
            currentConversationId = data.conversation_id; // Update conversation ID
            appendMessage(data.answer, 'model');
        } else {
            alert('Failed to get an answer.');
        }
    } catch (error) {
        alert('An error occurred while asking the PDF.');
        console.error('Ask PDF Error:', error);
    }
}

function appendMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    messageDiv.innerText = text;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll to bottom
}

async function fetchPdfs() {
    try {
        const response = await fetch('/get_pdfs', {
            headers: { 'Authorization': `Bearer ${accessToken}` }
        });
        if (response.ok) {
            const data = await response.json();
            renderPdfList(data.pdfs);
        } else {
            console.error('Failed to fetch PDFs');
        }
    } catch (error) {
        console.error('Error fetching PDFs:', error);
    }
}

function renderPdfList(pdfs) {
    pdfList.innerHTML = ''; // Clear existing list
    pdfs.forEach(pdf => {
        const li = document.createElement('li');
        li.textContent = pdf.filename;
        li.dataset.id = pdf._id;
        li.onclick = () => selectPdf(pdf._id, pdf.filename);
        pdfList.appendChild(li);
    });
}

function selectPdf(pdfId, filename) {
    currentPdfId = pdfId;
    currentConversationId = null; // Reset conversation when switching PDFs
    chatHistory.innerHTML = ''; // Clear chat history
    chatHeader.textContent = `Chat with: ${filename}`;

    // Highlight the selected PDF
    document.querySelectorAll('#pdf-list li').forEach(li => {
        li.classList.remove('selected');
        if (li.dataset.id === pdfId) {
            li.classList.add('selected');
        }
    });
}

