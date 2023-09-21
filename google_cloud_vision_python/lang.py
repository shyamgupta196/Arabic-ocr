## to translate to other languages ! RUN THIS. you can find language codes from here.

from google.cloud import translate

def get_supported_languages(
    project_id: str = "YOUR_PROJECT_ID",
) -> translate.SupportedLanguages:
    """Getting a list of supported language codes.

    Args:
        project_id: The GCP project ID.

    Returns:
        A list of supported language codes.
    """
    client = translate.TranslationServiceClient()

    parent = f"projects/{project_id}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    response = client.get_supported_languages(parent=parent)

    # List language codes of supported languages.
    print("Supported Languages:")
    for language in response.languages:
        print(f"Language Code: {language.language_code}")

    return response

print(get_supported_languages('gcp-kubernetes-ml-app'))