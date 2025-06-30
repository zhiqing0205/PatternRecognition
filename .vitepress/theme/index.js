import DefaultTheme from 'vitepress/theme'
import './custom.css'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    // MathJax 滚动修复
    if (typeof window !== 'undefined') {
      // 页面加载后修复 MathJax 滚动问题
      const fixMathJaxScroll = () => {
        const containers = document.querySelectorAll('mjx-container[display="true"]');
        containers.forEach(container => {
          container.style.overflow = 'visible';
          container.style.width = 'auto';
          container.style.maxWidth = 'none';
          container.style.height = 'auto';
          
          const mathElement = container.querySelector('mjx-math');
          if (mathElement) {
            mathElement.style.overflow = 'visible';
            mathElement.style.width = 'auto';
            mathElement.style.maxWidth = 'none';
            mathElement.style.height = 'auto';
          }
        });
      };

      // 监听路由变化和 MathJax 渲染完成
      const observer = new MutationObserver((mutations) => {
        let shouldFix = false;
        mutations.forEach((mutation) => {
          if (mutation.addedNodes.length > 0) {
            mutation.addedNodes.forEach((node) => {
              if (node.nodeType === 1 && 
                  (node.tagName === 'MJX-CONTAINER' || 
                   node.querySelector && node.querySelector('mjx-container[display="true"]'))) {
                shouldFix = true;
              }
            });
          }
        });
        if (shouldFix) {
          setTimeout(fixMathJaxScroll, 100);
        }
      });

      // 页面加载完成后开始监听
      window.addEventListener('load', () => {
        fixMathJaxScroll();
        observer.observe(document.body, {
          childList: true,
          subtree: true
        });
      });

      // 路由变化时也要修复
      window.addEventListener('popstate', () => {
        setTimeout(fixMathJaxScroll, 500);
      });
    }
  }
}