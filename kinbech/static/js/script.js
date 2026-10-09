// static/js/script.js
document.addEventListener("DOMContentLoaded", function () {
  // Update condition display
  const conditionInput = document.getElementById("condition");
  if (conditionInput) {
    conditionInput.addEventListener("input", function () {
      const value = this.value;
      let description = "";

      if (value <= 2) description = "🔴 Poor";
      else if (value <= 4) description = "🟠 Fair";
      else if (value <= 6) description = "🟡 Good";
      else if (value <= 8) description = "🟢 Very Good";
      else description = "🟢 Excellent";

      const display = document.getElementById("conditionValue");
      if (display) {
        display.textContent = `${value} - ${description}`;
      }
    });
  }

  // Form validation
  const form = document.getElementById("predictionForm");
  if (form) {
    form.addEventListener("submit", function (e) {
      const originalPrice = document.getElementById("original_price").value;
      const ageYears = document.getElementById("age_years").value;
      const category = document.getElementById("category").value;

      if (!originalPrice || !ageYears || !category) {
        e.preventDefault();
        alert("Please fill all required fields");
      }
    });
  }
});

// Fetch and display API data (optional, for real-time updates)
async function loadPriceStats() {
  try {
    const response = await fetch("/api/price-stats");
    const data = await response.json();
    console.log("Price Statistics:", data);
  } catch (error) {
    console.error("Error loading price stats:", error);
  }
}
