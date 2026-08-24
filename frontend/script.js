// ---------- Config ----------
// Set window.BACKEND_URL before this script to use a deployed API.
const BACKEND_URL = window.BACKEND_URL || "http://127.0.0.1:8000/chat";

// ---------- Grab elements from the page ----------
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

// ---------- Helper: add a message bubble to the chat box ----------
function addMessage(text, sender) {
  // sender is either "user" or "bot"
  const message = document.createElement("div");
  message.classList.add("message", sender === "user" ? "user-message" : "bot-message");
  message.textContent = text;
  chatBox.appendChild(message);

  // Auto-scroll to the newest message
  chatBox.scrollTop = chatBox.scrollHeight;

  return message;
}

// ---------- Main function: send the question to the backend ----------
async function sendMessage() {
  const question = userInput.value.trim();

  // Step 1: don't send empty questions
  if (question === "") {
    return;
  }

  // Step 2: show the user's message immediately (right side)
  addMessage(question, "user");

  // Step 3: clear the input field
  userInput.value = "";

  // Step 4: show a temporary "Thinking..." bubble while we wait
  const thinkingBubble = addMessage("Thinking...", "bot");

  try {
    // Step 5: send the question to FastAPI
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question: question })
    });

    if (!response.ok) {
      let detail = "Backend returned status " + response.status;
      try {
        const errorData = await response.json();
        detail = errorData.detail || detail;
      } catch (parseError) {
        // Keep the status message when the backend response is not JSON.
      }
      throw new Error(detail);
    }

    // Step 6: read the JSON response
    const data = await response.json();

    // Step 7: replace "Thinking..." with the real answer
    thinkingBubble.textContent = data.answer;

  } catch (error) {
    console.error(error);
    thinkingBubble.textContent = error.message === "Failed to fetch"
      ? "The chatbot service is not available yet. Please start or deploy the backend."
      : error.message || "Unable to connect to the chatbot backend.";
  }
}

// ---------- Event listeners ----------

// Send when the button is clicked
sendButton.addEventListener("click", sendMessage);

// Send when the Enter key is pressed inside the input field
userInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    sendMessage();
  }
});
