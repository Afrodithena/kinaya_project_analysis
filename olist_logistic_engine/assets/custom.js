/**
 * Olist Logistics Engine - Custom JavaScript
 * Additional interactivity and client-side features
 */

// Page load tracking
document.addEventListener('DOMContentLoaded', function() {
    console.log('Olist Logistics Intelligence Dashboard loaded');
    
    // Track page view
    if (typeof gtag !== 'undefined') {
        gtag('event', 'page_view', {
            'page_title': document.title,
            'page_location': window.location.href
        });
    }
    
    // Add animation to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.transition = 'all 0.2s ease';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Auto-refresh toggle for maps (optional)
    let autoRefresh = true;
    window.toggleAutoRefresh = function() {
        autoRefresh = !autoRefresh;
        console.log('Auto-refresh:', autoRefresh);
        return autoRefresh;
    };
});

// Function to format numbers with K/M/B suffixes
window.formatNumber = function(num) {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toString();
};

// Function to export current view data
window.exportViewData = function(data, filename) {
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'export.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl + R to refresh data
    if (e.ctrlKey && e.key === 'r') {
        e.preventDefault();
        console.log('Manual refresh triggered');
        // Streamlit will handle refresh via rerun
    }
});