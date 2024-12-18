import React from 'react';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from './components/Login';
import Chatbot from './components/Chatbot';

const App = () => {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/chatbot" element={<Chatbot />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
