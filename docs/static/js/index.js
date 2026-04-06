var PLACEHOLDER_IMAGE =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";

function loadDeferredImage(image) {
  if (!image || image.dataset.loaded === "true") {
    return;
  }

  if (image.dataset.src) {
    image.src = image.dataset.src;
  }

  image.dataset.loaded = "true";
}

function setupDeferredImages() {
  var deferredImages = Array.from(document.querySelectorAll("img[data-src]"));
  deferredImages.forEach(function(image) {
    if (!image.getAttribute("src")) {
      image.setAttribute("src", PLACEHOLDER_IMAGE);
    }
  });

  if (!("IntersectionObserver" in window)) {
    deferredImages.forEach(loadDeferredImage);
    return;
  }

  var observer = new IntersectionObserver(
    function(entries) {
      entries.forEach(function(entry) {
        if (!entry.isIntersecting) {
          return;
        }
        loadDeferredImage(entry.target);
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "300px 0px" }
  );

  deferredImages.forEach(function(image) {
    observer.observe(image);
  });
}

function setupForwardCarousel() {
  var forwardCarousel = document.querySelector(".forward-carousel");
  if (!forwardCarousel) {
    return;
  }

  var viewport = forwardCarousel.querySelector(".forward-carousel__viewport");
  var items = Array.from(forwardCarousel.querySelectorAll(".forward-carousel__item"));
  var dots = Array.from(forwardCarousel.querySelectorAll(".forward-carousel__dot"));
  var prevButton = forwardCarousel.querySelector(".forward-carousel__nav--prev");
  var nextButton = forwardCarousel.querySelector(".forward-carousel__nav--next");
  var activeIndex = 0;

  var updateItems = function(index) {
    items.forEach(function(item, itemIndex) {
      var isActive = itemIndex === index;
      item.classList.toggle("is-active", isActive);
      item.setAttribute("aria-hidden", (!isActive).toString());

      var image = item.querySelector("img[data-src]");
      if (isActive && image) {
        loadDeferredImage(image);
      }
    });

    dots.forEach(function(dot, dotIndex) {
      dot.classList.toggle("is-active", dotIndex === index);
      dot.setAttribute("aria-selected", (dotIndex === index).toString());
      dot.setAttribute("tabindex", dotIndex === index ? "0" : "-1");
    });

    forwardCarousel.setAttribute("data-active-index", index);
  };

  var goTo = function(index) {
    var total = items.length;
    if (total === 0) {
      return;
    }

    if (index < 0) {
      index = total - 1;
    } else if (index >= total) {
      index = 0;
    }

    activeIndex = index;
    updateItems(activeIndex);
  };

  if (prevButton) {
    prevButton.addEventListener("click", function() {
      goTo(activeIndex - 1);
    });
  }

  if (nextButton) {
    nextButton.addEventListener("click", function() {
      goTo(activeIndex + 1);
    });
  }

  dots.forEach(function(dot, index) {
    dot.addEventListener("click", function() {
      goTo(index);
    });
  });

  forwardCarousel.addEventListener("keydown", function(event) {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      goTo(activeIndex - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      goTo(activeIndex + 1);
    }
  });

  var pointerStartX = null;
  if (viewport) {
    viewport.addEventListener("pointerdown", function(event) {
      pointerStartX = event.clientX;
    });

    viewport.addEventListener("pointerup", function(event) {
      if (pointerStartX === null) {
        return;
      }

      var delta = event.clientX - pointerStartX;
      if (Math.abs(delta) > 40) {
        if (delta > 0) {
          goTo(activeIndex - 1);
        } else {
          goTo(activeIndex + 1);
        }
      }

      pointerStartX = null;
    });

    ["pointerleave", "pointercancel"].forEach(function(eventName) {
      viewport.addEventListener(eventName, function() {
        pointerStartX = null;
      });
    });
  }

  updateItems(activeIndex);
}

document.addEventListener("DOMContentLoaded", function() {
  setupDeferredImages();
  setupForwardCarousel();
});
