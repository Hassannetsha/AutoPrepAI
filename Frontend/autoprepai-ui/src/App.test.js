import { render, screen } from '@testing-library/react';
import App from './App';

test('renders AutoPrepAI app', () => {
  render(<App />);
  expect(screen.getAllByText(/AutoPrepAI/i).length).toBeGreaterThan(0);
});
