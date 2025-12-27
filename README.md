
# Edututor AI - Interactive Learning Assistant (Gradio Frontend)

This project demonstrates a basic interactive learning assistant built with Gradio, leveraging the `ibm-granite/granite-3.2-2b-instruct` model for AI functionalities.

## Project Overview

The primary purpose of this project is to showcase how to integrate and utilize a large language model (LLM) like Granite directly within a Google Colab environment for educational applications. It provides a simple Gradio-based frontend for user interaction.

## Current Features

- **Concept Explanation**: Users can input a concept, and the Granite model will provide a detailed explanation with examples.
- **Quiz Generation**: Users can request a multiple-choice quiz on a specific topic, and the model will generate 5 questions with options and correct answers.
- **Quiz Scoring**: Users can answer the generated quiz, and the application will provide a score and feedback.
- **User Role Selection**: Allows users to select if they are a 'Student' or 'Educator' at login.

## Current Limitations & Future Scope

It is important to note that this project is currently a frontend-only demonstration. As such, the following features are **yet to be included**:

- **Robust Backend Features**: All 'database' functionalities (`students_db`, `educators_db`, `dashboard_db`) are currently in-memory (runtime-only) Python dictionaries. This means data is not persisted across sessions.
- **Full Educator Dashboard**: While an 'Educator Dashboard' UI exists, its functionality is limited to fetching student progress from the in-memory `dashboard_db` for the current session. Comprehensive features like managing students, creating content, or viewing aggregated analytics are not yet implemented.
- **Persistent User Data**: User registrations (student/educator) and quiz results are not saved permanently.
- **Security & Authentication**: Advanced security measures, proper user authentication (beyond basic email input), and authorization are not part of this demonstration.

This project serves as a foundational step, highlighting the direct use of the Granite model for interactive educational content generation, with a clear path for future expansion into a full-fledged application.
