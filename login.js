document.addEventListener("DOMContentLoaded", function() {
    const loginForm = document.getElementById("login-form");

    loginForm.addEventListener("submit", function(event) {
        event.preventDefault(); // Prevents default form submission

        // Get the email and password input values
        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value.trim();

        // For demonstration purposes, log credentials to the console
        console.log("Email:", email);
        console.log("Password:", password);

        // Validate or send to server here, then redirect if login is successful
        // In this example, it will directly navigate to "index.html"
        window.location.href = "index.html";
    });
});
