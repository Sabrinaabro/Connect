"use client";

import { useState } from "react";

export default function HomePage() {
  const[loading, setLoading] = useState(false);
  const [response, setResponse] = useState<string | null>(null);
  
  const handleLetsTalk = async () => {
    setLoading(true);
    try{
        const res = await fetch("http://localhost:5000/start-gesture")
        const data = await res.json();
        setResponse(data.message);
    }
    catch(error){
        console.error("Error:", error);
        setResponse("Failed to start the model.");
    }finally{
        setLoading(false);
    }
    };
    return(
        <main className="flex flex-col items-center justify-center min-h-screen bg-green-100">
            <h2 className="text-xl font-bold text-blue-800">Connect</h2>
            <p className="text-lg text-gray-600 mt-4">
                Where Sign Language Meets Understanding.
            </p>
            <button
            onClick={handleLetsTalk}
            className="mt-6 px-6 py-3 bg-blue-600 text-white rounded-lg shadow-lg hover:bg-blue-700 transition"
            disabled={loading}
            >
                {loading ? "Starting... " : "Let's Talk"}
            </button>
            {response && <p className="mt-4 text-green-600">{response}</p>}
        </main>
    );
    }
