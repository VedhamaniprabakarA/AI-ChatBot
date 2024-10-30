import React, { useState, useEffect } from "react";
import '../styles.css';

const Chatbot = () => {
    const [csrfToken, setCsrfToken] = useState("");
    const [prompt, setPrompt] = useState("");
    const [responses, setResponses] = useState([]);
    const [darkMode, setDarkMode] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Fetch the CSRF token on mount
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

    // Handle form submission
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
                        "X-CSRFToken": csrfToken,  // Include CSRF token here
                    },
                    body: JSON.stringify({ question: prompt }),
                });

                const data = await response.json();
                if (response.ok) {
                    setResponses([...responses, { question: prompt, answer: data.response }]);
                    setPrompt(""); // Clear input after submission
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

    // Toggle dark/light mode
    const toggleMode = () => {
        setDarkMode(!darkMode);
    };

    return (
        <div className={`chatbot-container ${darkMode ? 'dark-mode' : 'light-mode'}`}>
            <div className="sidebar">
                <div className="profile-info">
                    <img src="/path-to-profile-pic.jpg" alt="Profile" className="profile-pic" />
                    <h3>NerdsHub</h3>
                    <p>Contact: nerdshub.co.in</p>
                </div>
                <button className="settings-btn">Settings</button>
                <button onClick={toggleMode} className="toggle-mode-btn">
                    {darkMode ? "Light Mode" : "Dark Mode"}
                </button>
            </div>

            <div className="chatbot-content">
                <h1>Welcome to AI_ChatBot</h1>
                <p>Your personalized chatbot awaits!</p>

                <div className="responses">
                    {responses.map((response, index) => (
                        <div key={index} className="response">
                            <p><strong>You:</strong> {response.question}</p>
                            <p><strong>DobbyBot:</strong> {response.answer}</p>
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
