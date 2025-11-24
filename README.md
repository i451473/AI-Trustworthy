# AI-Trustworthy

This is a C# .NET 8 console application project designed to process markdown reports and generate AI-assisted summaries with trust and consistency validation using Azure OpenAI API.

## Features

- Reads markdown file `report.md`
- Generates multiple AI-powered summaries with retries
- Compares summaries for consistency using embeddings
- Validates summary sentences factually against the source
- Outputs a trust-enhanced HTML report with results

## Requirements

- .NET 8 SDK
- Azure OpenAI API credentials in User Secrets (`AzureOpenAI:Endpoint`, `AzureOpenAI:Key`, etc.)

## Setup

1. Clone or download the project.
2. Open the solution `AI-Trustworthy.sln` in Visual Studio 2022 or later.
3. Configure your Azure OpenAI settings in User Secrets.
4. Place the markdown report file as `report.md` in the app working directory.
5. Run the console application.

## Notes

This project uses these NuGet packages:

- Azure.AI.OpenAI
- Microsoft.Extensions.AI
- Microsoft.Extensions.Configuration
- OpenAI.Embeddings

## License

This project is provided as-is without any warranty.
