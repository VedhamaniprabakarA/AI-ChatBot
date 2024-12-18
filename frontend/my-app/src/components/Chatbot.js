import React, { useState, useEffect } from "react";
import '../styles.css';
/*........................*/
const Chatbot = () => {
    const [csrfToken, setCsrfToken] = useState("");
    const [prompt, setPrompt] = useState("");
    const [responses, setResponses] = useState([]);
    const [darkMode, setDarkMode] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedModel, setSelectedModel] = useState("gpt-2");

    useEffect(() => {
        const fetchCsrfToken = async () => {
            try {
                const response = await fetch("http://localhost:8000/api/csrf/");
                const data = await response.json();
                if (response.ok) {
                    setCsrfToken(data.csrfToken);
                } else {
                    console.error("Failed to fetch CSRF token");
                }
            } catch (error) {
                console.error("Error fetching CSRF token:", error);
            }
        };
        fetchCsrfToken();
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (prompt.trim()) {
            setLoading(true);
            setError(null);
            try {
                const response = await fetch("http://localhost:8000/api/ask/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": csrfToken,
                    },
                    body: JSON.stringify({ question: prompt, model: selectedModel }),
                });

                const data = await response.json();
                if (response.ok) {
                    setResponses([...responses, { question: prompt, answer: data.response }]);
                    setPrompt("");
                } else {
                    setError(data.error || "An error occurred. Please try again.");
                }
            } catch (error) {
                setError("Error connecting to the chatbot. Please try again later.");
                console.error("Error:", error);
            } finally {
                setLoading(false);
            }
        }
    };

    const toggleMode = () => {
        setDarkMode(!darkMode);
    };

    const handleModelChange = (e) => {
        setSelectedModel(e.target.value);
    };

    /*ChatBot text test*/

    return (
        <div className={`chatbot-container ${darkMode ? 'dark-mode' : 'light-mode'}`}>
            <div className="sidebar">
                <div className="profile-info">
                    <img src="profile.jpg" alt="Profile" className="profile-pic" />
                    <h3>NerdsHub</h3>
                    <p>AI Chatbot</p>
                </div>
                <div className="models-dropdown">
                    <button className="models-btn">Models</button>
                    <select value={selectedModel} onChange={handleModelChange} className="model-select">
                        <option value="gpt-2">GPT-2</option>
                        <option value="llama3.2:3b">Ollama</option>
                    </select>
                </div>
                <button className="settings-btn">Settings</button>
                <button onClick={toggleMode} className="toggle-mode-btn">
                    {darkMode ? "Light Mode" : "Dark Mode"}
                </button>
            </div>

            <div className="chatbot-content">
                <h1>Welcome to AI Chatbot</h1>
                <p>Your personalized chatbot awaits!</p>

                <div className="responses">
                    {responses.map((response, index) => (
                        <div key={index} className="response">
                            <p><strong>You:</strong> {response.question}</p>
                            <p><strong>EVE:</strong> {response.answer}</p>
                        </div>  
                    ))}
                    {loading && <p>Loading...</p>}
                    {error && <p className="error-message">{error}</p>}
                </div>

                <div className="partition"></div>

                <form onSubmit={handleSubmit} className="chatbot-form">
                    <input
                        type="text"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Type your message here..."
                        required
                        className="prompt-input"
                    />
                    <button type="submit" className="submit-btn" disabled={loading}>
                        {loading ? "Submitting..." : "Submit"}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default Chatbot;
