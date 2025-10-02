# Frontend Testing Documentation

This document provides an overview of the comprehensive frontend testing suite implemented for the NFL Sentiment Analyzer application.

## Test Structure

### Component Tests
- **SentimentDashboard.test.tsx**: Tests the main dashboard component including data loading, WebSocket connections, team filtering, and error handling
- **TeamSentimentCard.test.tsx**: Tests individual team sentiment cards with different sentiment states, trends, and user interactions
- **RealTimeChart.test.tsx**: Tests the chart component with various data states, mobile responsiveness, and team filtering
- **GamePredictionPanel.test.tsx**: Tests the game predictions panel with tab switching, data display, and empty states
- **Navbar.test.tsx**: Tests navigation component with authentication states and accessibility
- **LoadingSpinner.test.tsx**: Tests the loading spinner component with different sizes and states

### Hook Tests
- **useWebSocket.test.ts**: Tests WebSocket connection management, reconnection logic, message handling, and error scenarios
- **useSentimentWebSocket.test.ts**: Tests sentiment-specific WebSocket functionality, data processing, and real-time updates

### Page Tests
- **Login.test.tsx**: Tests login form validation, submission, loading states, and accessibility

### Integration Tests
- **RealTimeUpdates.test.tsx**: Tests end-to-end real-time data flow from WebSocket to UI components

### Accessibility Tests
- **accessibility.test.tsx**: Comprehensive accessibility testing using jest-axe, including:
  - WCAG compliance validation
  - Screen reader support
  - Keyboard navigation
  - Color contrast
  - Semantic HTML structure
  - ARIA attributes

### Visual Regression Tests
- **visual-regression.test.tsx**: Snapshot testing for UI consistency across different states and viewports

## Testing Features Covered

### 1. Unit Testing
- Component rendering and props handling
- State management and user interactions
- Form validation and submission
- Error boundary testing
- Loading states and empty states

### 2. Real-Time Data Updates
- WebSocket connection establishment
- Message handling and data processing
- Connection loss and reconnection
- Real-time UI updates
- Data synchronization

### 3. Accessibility Testing
- Screen reader compatibility
- Keyboard navigation support
- ARIA labels and descriptions
- Color contrast validation
- Semantic HTML structure
- Focus management

### 4. Visual Regression Testing
- Component appearance consistency
- Responsive design validation
- State-based visual changes
- Cross-browser compatibility
- Mobile viewport testing

### 5. Integration Testing
- End-to-end user workflows
- Component interaction testing
- API integration testing
- WebSocket integration testing
- Error handling across components

## Test Utilities

### Custom Render Function
The `render` function in `test/utils.tsx` provides:
- React Router context
- React Query client setup
- Authentication context
- Mock data generators

### Mock Data
Standardized mock data generators for:
- User objects
- Team data
- Game information
- Sentiment data
- WebSocket messages

### Accessibility Helpers
- jest-axe integration for automated accessibility testing
- Custom accessibility matchers
- Screen reader testing utilities

## Running Tests

```bash
# Run all tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm run test SentimentDashboard.test.tsx

# Run accessibility tests only
npm run test accessibility.test.tsx
```

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage for critical components
- **Integration Tests**: Cover main user workflows
- **Accessibility Tests**: 100% WCAG AA compliance
- **Visual Regression**: Key UI states and responsive breakpoints

## Best Practices

### 1. Test Structure
- Use descriptive test names
- Group related tests with `describe` blocks
- Follow AAA pattern (Arrange, Act, Assert)
- Clean up after each test

### 2. Accessibility Testing
- Test with screen readers in mind
- Validate keyboard navigation
- Check color contrast ratios
- Ensure proper ARIA usage

### 3. Real-Time Testing
- Mock WebSocket connections properly
- Test connection states and errors
- Validate data synchronization
- Test reconnection scenarios

### 4. Visual Testing
- Use consistent viewport sizes
- Test responsive breakpoints
- Validate loading and error states
- Check component interactions

## Continuous Integration

Tests are configured to run automatically on:
- Pull request creation
- Code commits to main branch
- Scheduled nightly runs

## Troubleshooting

### Common Issues
1. **WebSocket Mock Issues**: Ensure proper cleanup of WebSocket instances
2. **Async Test Failures**: Use `waitFor` for async operations
3. **Accessibility Violations**: Check ARIA labels and semantic HTML
4. **Visual Regression Failures**: Update snapshots when UI changes are intentional

### Debug Tips
- Use `screen.debug()` to inspect rendered DOM
- Add `data-testid` attributes for reliable element selection
- Use React DevTools for component state inspection
- Check browser console for accessibility warnings

## Future Enhancements

1. **Performance Testing**: Add tests for component rendering performance
2. **E2E Testing**: Implement Cypress tests for full user journeys
3. **Cross-Browser Testing**: Add automated testing across different browsers
4. **Mobile Testing**: Expand mobile-specific test coverage
5. **Internationalization**: Add tests for multi-language support