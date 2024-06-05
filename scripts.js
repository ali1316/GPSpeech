document.addEventListener('DOMContentLoaded', function() {
    const micButton = document.getElementById('mic-button');
    const messagesList = document.getElementById('messages');
    let recognition;

    // Check if SpeechRecognition is supported
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false; 
        recognition.interimResults = false; 
        recognition.lang = 'en-US'; 

        recognition.onstart = function() {
            console.log("Voice recording started...");
            micButton.classList.add('recording'); 
        };

        recognition.onend = function() {
            console.log("Voice recording ended.");
            micButton.classList.remove('recording'); 
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            addMessage(transcript, 'user');

            // Here, provide a placeholder response or integrate with your own backend
            const botResponse = `You said: "${transcript}". This is a generated response.`;
            setTimeout(() => {
                addMessage(botResponse, 'bot');
            }, 1000); 
        };
    } else {
        alert("Speech Recognition API is not supported in this browser.");
    }

    // Add a message to the UI
    const addMessage = (text, type) => {
        const messageItem = document.createElement('li');
        messageItem.className = type;
        messageItem.textContent = text;
        messagesList.appendChild(messageItem);
        messagesList.scrollTop = messagesList.scrollHeight;
    };

    // Trigger voice recognition on button click
    micButton.addEventListener('click', function() {
        if (recognition) {
            recognition.start();
        }
    });
});
