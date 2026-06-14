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

let agentModeActive = false;
let currentSessionId = null;
let currentStep = null;
let originalQueryText = ""; // To track the original user query for fallback

// --- DOM ELEMENTS ---
const authContainer = document.getElementById('auth-container');
const appContainer = document.getElementById('app-container');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const chatHistory = document.getElementById('chat-history');
const pdfList = document.getElementById('pdf-list');
const chatHeader = document.getElementById('chat-header');

const agentConsole = document.getElementById('agent-console-container');
const agentLogs = document.getElementById('agent-logs');
const agentControls = document.getElementById('agent-controls');
const statusIndicator = document.getElementById('agent-status-indicator');

// --- UI TOGGLING ---
function toggleAuthForms() {
    loginForm.style.display = loginForm.style.display === 'none' ? 'block' : 'none';
    registerForm.style.display = registerForm.style.display === 'none' ? 'block' : 'none';
}

function showApp() {
    authContainer.style.display = 'none';
    appContainer.style.display = 'grid'; // Grid display for dashboard
    
    // Display header user info
    document.getElementById('header-user-actions').style.display = 'flex';
    
    // Check JWT payload to get username if needed (simple display)
    try {
        const payload = JSON.parse(atob(accessToken.split('.')[1]));
        document.getElementById('user-display').innerHTML = `Logged in as: <strong>${payload.sub}</strong>`;
    } catch(e) {
        document.getElementById('user-display').innerHTML = `Logged in as: <strong>User</strong>`;
    }
}

function showAuth() {
    sessionStorage.removeItem('accessToken');
    accessToken = null;
    authContainer.style.display = 'block';
    appContainer.style.display = 'none';
    document.getElementById('header-user-actions').style.display = 'none';
}

function logout() {
    showAuth();
}

function updateUploadLabel() {
    const fileInput = document.getElementById('pdf-file-input');
    const label = document.getElementById('upload-label');
    if (fileInput.files.length > 0) {
        label.textContent = `Selected: ${fileInput.files[0].name}`;
    } else {
        label.textContent = "Drag & drop or click to upload PDF";
    }
}

function toggleAgentModeOptions() {
    const checkbox = document.getElementById('agent-mode-checkbox');
    agentModeActive = checkbox.checked;
    
    if (agentModeActive) {
        appContainer.classList.add('layout-3col');
        agentConsole.style.display = 'flex';
        logConsole('system', 'Agentic RAG Mode enabled. HITL gates ready.');
    } else {
        appContainer.classList.remove('layout-3col');
        agentConsole.style.display = 'none';
    }
}

// Console Logging Helper
function logConsole(type, message) {
    const div = document.createElement('div');
    div.classList.add('console-line', `${type}-line`);
    
    const timestamp = new Date().toLocaleTimeString();
    div.textContent = `[${timestamp}] ${message}`;
    
    agentLogs.appendChild(div);
    agentLogs.scrollTop = agentLogs.scrollHeight;
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
            
            // Reset upload field label
            pdfFileInput.value = '';
            updateUploadLabel();
        } else {
            alert('PDF upload failed.');
        }
    } catch (error) {
        alert('An error occurred during PDF upload.');
        console.error('Upload Error:', error);
    }
}

function handleQueryKey(event) {
    if (event.key === 'Enter') {
        submitQuery();
    }
}

async function submitQuery() {
    const userQueryInput = document.getElementById('user-query');
    const userQuery = userQueryInput.value.strip ? userQueryInput.value.strip() : userQueryInput.value.trim();
    
    if (!userQuery) {
        return;
    }
    
    if (!currentPdfId) {
        alert('Please select a PDF document first.');
        return;
    }
    
    originalQueryText = userQuery;
    appendMessage(userQuery, 'user');
    userQueryInput.value = '';
    
    if (!agentModeActive) {
        // Standard RAG execution
        askPdfStandard(userQuery);
    } else {
        // Agentic RAG execution
        startAgentWorkflow(userQuery);
    }
}

// Standard RAG Route Execution
async function askPdfStandard(userQuery) {
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
            const errData = await response.json();
            appendMessage(`Error: ${errData.msg || 'Failed to get an answer.'}`, 'model');
        }
    } catch (error) {
        console.error('Standard RAG Error:', error);
        appendMessage('An error occurred while communicating with the server.', 'model');
    }
}

// --- AGENTIC RAG METHOD CALLS ---

async function startAgentWorkflow(userQuery) {
    setAgentStatus('active', 'Working');
    agentControls.style.display = 'none';
    agentControls.innerHTML = '';
    
    agentLogs.innerHTML = '';
    logConsole('system', 'Starting Agentic RAG workflow...');
    logConsole('agent', `User Query: "${userQuery}"`);
    
    const hitlDecomp = document.getElementById('hitl-decomp-checkbox').checked;
    const hitlWeb = document.getElementById('hitl-web-checkbox').checked;
    
    try {
        const response = await fetch('/agent/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${accessToken}`
            },
            body: JSON.stringify({
                query: userQuery,
                pdf_id: currentPdfId,
                conversation_id: currentConversationId,
                options: {
                    hitl_decomposer: hitlDecomp,
                    hitl_web: hitlWeb
                }
            })
        });

        if (response.ok) {
            const data = await response.json();
            handleAgentResponse(data);
        } else {
            const errData = await response.json();
            logConsole('error', `Failed to start workflow: ${errData.message || 'Server error'}`);
            setAgentStatus('idle', 'Error');
            appendMessage('Agent execution failed. See logs for details.', 'model');
        }
    } catch (error) {
        console.error('Agent Start Error:', error);
        logConsole('error', `Exception: ${error.message}`);
        setAgentStatus('idle', 'Error');
    }
}

function handleAgentResponse(data) {
    currentSessionId = data.session_id;
    
    // Append logs
    if (data.logs && data.logs.length > 0) {
        agentLogs.innerHTML = '';
        data.logs.forEach(log => {
            let type = 'info';
            if (log.toLowerCase().includes('waiting for') || log.toLowerCase().includes('decision:')) {
                type = 'warning';
            } else if (log.toLowerCase().includes('successfully') || log.toLowerCase().includes('completed')) {
                type = 'success';
            } else if (log.toLowerCase().includes('warning') || log.toLowerCase().includes('error')) {
                type = 'error';
            } else if (log.startsWith('Detecting') || log.startsWith('Decomposing') || log.startsWith('Running web')) {
                type = 'agent';
            }
            logConsole(type, log);
        });
    }

    if (data.status === 'completed') {
        setAgentStatus('idle', 'Idle');
        currentConversationId = data.conversation_id;
        appendMessage(data.answer, 'model');
        agentControls.style.display = 'none';
        agentControls.innerHTML = '';
    } else if (data.status === 'pending_approval') {
        setAgentStatus('waiting', 'Requires User Approval');
        currentStep = data.step;
        renderHITLControls(data);
    }
}

function setAgentStatus(status, text) {
    statusIndicator.className = `pulse-indicator status-${status}`;
    statusIndicator.textContent = text;
}

function renderHITLControls(data) {
    agentControls.innerHTML = '';
    agentControls.style.display = 'block';
    
    const container = document.createElement('div');
    
    if (data.step === 'query_decomposition') {
        const title = document.createElement('h3');
        title.textContent = 'Query Decomposer HITL';
        const desc = document.createElement('p');
        desc.textContent = 'The agent decomposed your query into these sub-queries. You can add, edit, or remove queries before they search the PDF document:';
        
        container.appendChild(title);
        container.appendChild(desc);
        
        const listDiv = document.createElement('div');
        listDiv.className = 'query-edit-list';
        listDiv.id = 'decomposed-queries-list';
        
        const queries = data.data.decomposed_queries || [];
        queries.forEach((q, idx) => {
            listDiv.appendChild(createQueryRow(q, idx));
        });
        container.appendChild(listDiv);
        
        // Add Query Button
        const addBtn = document.createElement('button');
        addBtn.className = 'btn-add-query';
        addBtn.textContent = '+ Add Sub-query';
        addBtn.onclick = () => {
            const rows = document.getElementById('decomposed-queries-list');
            rows.appendChild(createQueryRow('', rows.children.length));
        };
        container.appendChild(addBtn);
        
        // Action Buttons Row
        const actionsRow = document.createElement('div');
        actionsRow.className = 'control-actions-row';
        
        const approveBtn = document.createElement('button');
        approveBtn.className = 'btn-approve';
        approveBtn.textContent = 'Approve & Search PDF';
        approveBtn.onclick = () => submitDecomposerApproval();
        
        const skipBtn = document.createElement('button');
        skipBtn.className = 'btn-skip';
        skipBtn.textContent = 'Skip / Use Original';
        skipBtn.onclick = () => skipDecomposerApproval();
        
        actionsRow.appendChild(approveBtn);
        actionsRow.appendChild(skipBtn);
        container.appendChild(actionsRow);
    } 
    else if (data.step === 'web_search') {
        const title = document.createElement('h3');
        title.textContent = 'Web Search HITL';
        const desc = document.createElement('p');
        desc.textContent = 'The agent determined that the PDF context is incomplete and wishes to search the web. You can edit the web search queries below:';
        
        container.appendChild(title);
        container.appendChild(desc);
        
        const listDiv = document.createElement('div');
        listDiv.className = 'query-edit-list';
        listDiv.id = 'web-queries-list';
        
        const queries = data.data.web_queries || [];
        queries.forEach((q, idx) => {
            listDiv.appendChild(createQueryRow(q, idx));
        });
        container.appendChild(listDiv);
        
        // Add Query Button
        const addBtn = document.createElement('button');
        addBtn.className = 'btn-add-query';
        addBtn.textContent = '+ Add Web Query';
        addBtn.onclick = () => {
            const rows = document.getElementById('web-queries-list');
            rows.appendChild(createQueryRow('', rows.children.length));
        };
        container.appendChild(addBtn);
        
        // Action Buttons Row
        const actionsRow = document.createElement('div');
        actionsRow.className = 'control-actions-row';
        
        const approveBtn = document.createElement('button');
        approveBtn.className = 'btn-approve';
        approveBtn.textContent = 'Approve & Search Web';
        approveBtn.onclick = () => submitWebApproval();
        
        const skipBtn = document.createElement('button');
        skipBtn.className = 'btn-skip';
        skipBtn.textContent = 'Skip Web Search';
        skipBtn.onclick = () => skipWebApproval();
        
        actionsRow.appendChild(approveBtn);
        actionsRow.appendChild(skipBtn);
        container.appendChild(actionsRow);
    }
    
    agentControls.appendChild(container);
    agentControls.scrollIntoView({ behavior: 'smooth' });
}

function createQueryRow(val, idx) {
    const row = document.createElement('div');
    row.className = 'query-edit-row';
    
    const input = document.createElement('input');
    input.type = 'text';
    input.value = val;
    input.className = 'query-input-field';
    
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn-remove-query';
    removeBtn.textContent = '×';
    removeBtn.onclick = () => row.remove();
    
    row.appendChild(input);
    row.appendChild(removeBtn);
    return row;
}

// Approve & Skip Handler functions for Decomposer
async function submitDecomposerApproval() {
    const inputElements = document.querySelectorAll('#decomposed-queries-list .query-input-field');
    const queries = [];
    inputElements.forEach(input => {
        const val = input.value.trim();
        if (val) queries.push(val);
    });
    
    if (queries.length === 0) {
        alert('Please provide at least one query, or click Skip.');
        return;
    }
    
    setAgentStatus('active', 'Working');
    agentControls.style.display = 'none';
    logConsole('system', 'Sending approved sub-queries to agent...');
    
    sendStepApproval('query_decomposition', { decomposed_queries: queries });
}

async function skipDecomposerApproval() {
    setAgentStatus('active', 'Working');
    agentControls.style.display = 'none';
    logConsole('system', 'Skipping decomposer. Using original query.');
    
    sendStepApproval('query_decomposition', { decomposed_queries: [originalQueryText] });
}

// Approve & Skip Handler functions for Web Search
async function submitWebApproval() {
    const inputElements = document.querySelectorAll('#web-queries-list .query-input-field');
    const queries = [];
    inputElements.forEach(input => {
        const val = input.value.trim();
        if (val) queries.push(val);
    });
    
    if (queries.length === 0) {
        alert('Please provide at least one search query, or click Skip.');
        return;
    }
    
    setAgentStatus('active', 'Working');
    agentControls.style.display = 'none';
    logConsole('system', 'Sending approved web search queries to agent...');
    
    sendStepApproval('web_search', { web_queries: queries });
}

async function skipWebApproval() {
    setAgentStatus('active', 'Working');
    agentControls.style.display = 'none';
    logConsole('system', 'Skipping web search. Synthesizing answer based only on PDF.');
    
    sendStepApproval('web_search', { web_queries: [] });
}

// Generic API caller for approval
async function sendStepApproval(step, dataPayload) {
    try {
        const response = await fetch('/agent/approve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${accessToken}`
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                step: step,
                data: dataPayload
            })
        });

        if (response.ok) {
            const resData = await response.json();
            handleAgentResponse(resData);
        } else {
            const errData = await response.json();
            logConsole('error', `Failed to approve step: ${errData.message || 'Server error'}`);
            setAgentStatus('idle', 'Error');
            appendMessage('Agent approval execution failed.', 'model');
        }
    } catch (error) {
        console.error('Agent Approve Error:', error);
        logConsole('error', `Exception: ${error.message}`);
        setAgentStatus('idle', 'Error');
    }
}

// Standard chat styling helper
function appendMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    
    // Formatting newlines as HTML line breaks, bold markdown, and links
    let formattedText = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Simple regex to link URLs
        .replace(/\[(.*?)\]\((https?:\/\/.*?)\)/g, '<a href="$2" target="_blank" class="chat-link">$1</a>')
        .replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" class="chat-link">$1</a>');
        
    messageDiv.innerHTML = formattedText;
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
    if (pdfs.length === 0) {
        const li = document.createElement('li');
        li.textContent = "No documents found.";
        li.style.cursor = "default";
        li.style.color = "var(--text-muted)";
        pdfList.appendChild(li);
        return;
    }
    
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
    chatHeader.textContent = `Chat: ${filename}`;
    
    // Reset agent console
    agentLogs.innerHTML = '<div class="console-line system-line">[System] New document loaded. Ready.</div>';
    agentControls.style.display = 'none';
    setAgentStatus('idle', 'Idle');

    // Highlight the selected PDF
    document.querySelectorAll('#pdf-list li').forEach(li => {
        li.classList.remove('selected');
        if (li.dataset.id === pdfId) {
            li.classList.add('selected');
        }
    });
}
